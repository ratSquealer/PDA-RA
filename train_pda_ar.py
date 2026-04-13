import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from curriculum import LearningSortingSampler
from datasets_pda_ar import PairedAugImageFolder
from losses import PDAARLoss
from modelyuan import CNNModel
from night_aug import NightAugTransform


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dann_alpha(step, total_steps):
    p = step / max(1, total_steps)
    return float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


def build_transforms(image_size):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    src = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    tgt = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NightAugTransform(3407),
            normalize,
        ]
    )
    return src, tgt


def train(args):
    seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    t_src, t_tgt = build_transforms(args.image_size)
    ds = PairedAugImageFolder(args.data_root, transform_source=t_src, transform_target=t_tgt)
    sampler = LearningSortingSampler(
        sorted_indices=ds.sorted_indices,
        dataset_size=len(ds),
        start_ratio=0.4,
        increment=0.15,
        step_epochs=10,
        seed=args.seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    model = CNNModel(num_classes=args.num_classes, backbone_pretrained=True, backbone_ckpt_path=args.backbone_ckpt)
    model.to(device)

    loss_fn = PDAARLoss(
        lambda_msf=1.0,
        lambda_ca=0.1,
        lambda_aa=0.01,
        lambda_p=0.1,
        lambda_domain=args.lambda_domain,
        label_smoothing_eps=0.1,
        pca_variance_ratio=0.95,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    global_step = 0
    total_steps = args.epochs * len(dl)

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)

        if epoch < args.warmup_epochs:
            warm_ratio = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warm_ratio
        else:
            cosine.step()

        running = {}
        for img_s, img_t, y in dl:
            img_s = img_s.to(device, non_blocking=True)
            img_t = img_t.to(device, non_blocking=True)
            y = torch.as_tensor(y, dtype=torch.long, device=device)

            alpha = dann_alpha(global_step, total_steps)
            optimizer.zero_grad(set_to_none=True)

            out_s = model(img_s, alpha, return_cam=True)
            out_t = model(img_t, alpha, return_cam=True)

            logits_s, domain_s, x1_s, x2_s, x3_s, x4_s, x5_s, fmap_s = out_s
            logits_t, domain_t, x1_t, x2_t, x3_t, x4_t, x5_t, fmap_t = out_t

            fc_weight = model.class_classifier.weight
            parts = loss_fn(
                logits_s=logits_s,
                logits_t=logits_t,
                domain_logits_s=domain_s,
                domain_logits_t=domain_t,
                feats_s=[x1_s, x2_s, x3_s, x4_s],
                feats_t=[x1_t, x2_t, x3_t, x4_t],
                pooled_s=x5_s,
                pooled_t=x5_t,
                feature_map_s=fmap_s,
                feature_map_t=fmap_t,
                fc_weight=fc_weight,
                labels=y,
            )

            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())

            global_step += 1

        denom = max(1, len(dl))
        log = {k: v / denom for k, v in running.items()}
        print(f"epoch {epoch+1}/{args.epochs} " + " ".join([f"{k}={log[k]:.4f}" for k in sorted(log.keys())]))

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.save_path)


def smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=8, backbone_pretrained=False).to(device)
    x = torch.randn(4, 3, 100, 100, device=device)
    logits, domain, x1, x2, x3, x4, x5, fmap = model(x, alpha=0.5, return_cam=True)
    loss_fn = PDAARLoss(lambda_domain=1.0)
    y = torch.randint(0, 8, (4,), device=device)
    parts = loss_fn(
        logits_s=logits,
        logits_t=logits,
        domain_logits_s=domain,
        domain_logits_t=domain,
        feats_s=[x1, x2, x3, x4],
        feats_t=[x1, x2, x3, x4],
        pooled_s=x5,
        pooled_t=x5,
        feature_map_s=fmap,
        feature_map_t=fmap,
        fc_weight=model.class_classifier.weight,
        labels=y,
    )
    parts["total"].backward()
    print("smoke ok", {k: float(v.detach().cpu()) for k, v in parts.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=100)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--backbone_ckpt", type=str, default=None)
    p.add_argument("--lambda_domain", type=float, default=1.0)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        smoke_test()
        return
    if not args.data_root:
        raise SystemExit("--data_root is required unless --smoke is set")
    train(args)


if __name__ == "__main__":
    main()

