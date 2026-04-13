import torch
import torch.nn.functional as F

from MMDloss import MMDLoss
from ssim import ssim_loss


def label_smoothing_cross_entropy(logits, targets, eps=0.1):
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(eps / num_classes)
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - eps + eps / num_classes)
    return (-true_dist * log_probs).sum(dim=1).mean()


def _svd(features):
    if hasattr(torch.linalg, "svd"):
        return torch.linalg.svd(features, full_matrices=False)
    u, s, v = torch.svd(features)
    return u, s, v.t()


def batch_pca(features, variance_ratio=0.95):
    centered = features - features.mean(dim=0, keepdim=True)
    u, s, vh = _svd(centered)
    var = s.pow(2)
    total = var.sum().clamp_min(1e-12)
    cum = torch.cumsum(var, dim=0) / total
    k = int((cum < variance_ratio).sum().item()) + 1
    k = min(k, centered.size(1))
    w = vh[:k].t().contiguous()
    z = centered @ w
    return z, w


def cam_from_feature_map(feature_map, fc_weight, class_idx):
    b, c, h, w = feature_map.shape
    w_cls = fc_weight.index_select(0, class_idx).view(b, c, 1, 1)
    cam = (feature_map * w_cls).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam_flat = cam.view(b, -1)
    cam_min = cam_flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    cam_max = cam_flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-12)
    return cam


def msfa_loss(mmd_fn, feats_s, feats_t):
    loss = 0.0
    for fs, ft in zip(feats_s, feats_t):
        loss = loss + mmd_fn(fs, ft)
    return loss


def pa_loss(mmd_fn, logits_s, logits_t):
    ps = F.softmax(logits_s, dim=1)
    pt = F.softmax(logits_t, dim=1)
    return mmd_fn(ps, pt)


def caa_loss(mmd_fn, feat_s, feat_t, labels, variance_ratio=0.95):
    if feat_s.ndim != 2 or feat_t.ndim != 2:
        raise ValueError("CAA expects 2D features [B, D]")
    if labels.ndim != 1:
        raise ValueError("labels must be [B]")

    combined = torch.cat([feat_s, feat_t], dim=0)
    z, w = batch_pca(combined, variance_ratio=variance_ratio)
    z_s = z[: feat_s.size(0)]
    z_t = z[feat_s.size(0) :]

    loss_intra = 0.0
    centers_s = []
    centers_t = []

    unique = torch.unique(labels)
    for cls in unique:
        mask = labels == cls
        zs = z_s[mask]
        zt = z_t[mask]
        if zs.numel() == 0 or zt.numel() == 0:
            continue
        cs = zs.mean(dim=0, keepdim=True)
        ct = zt.mean(dim=0, keepdim=True)
        loss_intra = loss_intra + (zs - cs).pow(2).sum(dim=1).mean()
        loss_intra = loss_intra + (zt - ct).pow(2).sum(dim=1).mean()
        centers_s.append(cs)
        centers_t.append(ct)

    if len(centers_s) == 0:
        return z_s.new_tensor(0.0)

    centers_s = torch.cat(centers_s, dim=0)
    centers_t = torch.cat(centers_t, dim=0)
    loss_center = mmd_fn(centers_s, centers_t)
    return loss_intra + loss_center


def aaa_loss(feature_map_s, feature_map_t, fc_weight, labels):
    cam_s = cam_from_feature_map(feature_map_s, fc_weight, labels)
    cam_t = cam_from_feature_map(feature_map_t, fc_weight, labels)
    return ssim_loss(cam_s, cam_t)


class PDAARLoss:
    def __init__(
        self,
        lambda_msf=1.0,
        lambda_ca=0.1,
        lambda_aa=0.01,
        lambda_p=0.1,
        lambda_domain=1.0,
        label_smoothing_eps=0.1,
        pca_variance_ratio=0.95,
    ):
        self.lambda_msf = float(lambda_msf)
        self.lambda_ca = float(lambda_ca)
        self.lambda_aa = float(lambda_aa)
        self.lambda_p = float(lambda_p)
        self.lambda_domain = float(lambda_domain)
        self.label_smoothing_eps = float(label_smoothing_eps)
        self.pca_variance_ratio = float(pca_variance_ratio)
        self.mmd = MMDLoss()

    def __call__(
        self,
        logits_s,
        logits_t,
        domain_logits_s,
        domain_logits_t,
        feats_s,
        feats_t,
        pooled_s,
        pooled_t,
        feature_map_s,
        feature_map_t,
        fc_weight,
        labels,
    ):
        l_label = label_smoothing_cross_entropy(logits_s, labels, eps=self.label_smoothing_eps) + label_smoothing_cross_entropy(
            logits_t, labels, eps=self.label_smoothing_eps
        )

        domain_labels_s = torch.zeros(domain_logits_s.size(0), dtype=torch.long, device=domain_logits_s.device)
        domain_labels_t = torch.ones(domain_logits_t.size(0), dtype=torch.long, device=domain_logits_t.device)
        l_domain = F.cross_entropy(domain_logits_s, domain_labels_s) + F.cross_entropy(domain_logits_t, domain_labels_t)

        l_msf = msfa_loss(self.mmd, feats_s, feats_t)
        l_ca = caa_loss(self.mmd, pooled_s, pooled_t, labels, variance_ratio=self.pca_variance_ratio)
        l_aa = aaa_loss(feature_map_s, feature_map_t, fc_weight, labels)
        l_p = pa_loss(self.mmd, logits_s, logits_t)

        total = (
            l_label
            + self.lambda_domain * l_domain
            + self.lambda_msf * l_msf
            + self.lambda_ca * l_ca
            + self.lambda_aa * l_aa
            + self.lambda_p * l_p
        )
        parts = {
            "total": total,
            "label": l_label,
            "domain": l_domain,
            "msf": l_msf,
            "ca": l_ca,
            "aa": l_aa,
            "p": l_p,
        }
        return parts
