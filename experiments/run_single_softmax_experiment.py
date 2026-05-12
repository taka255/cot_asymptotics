import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = os.path.join(ROOT_DIR, "experiments", "results", "single_softmax")
EVAL_T_LIST = list(range(0, 32))
ATTN_MASK_MODE = "data_only"
USE_HISTORY_AT_INFERENCE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_task_batch(batch_size, n, d, device):
    x = torch.randn(batch_size, d, n, device=device)
    w_star = torch.randn(batch_size, d, device=device)
    y = torch.einsum("bd,bdn->bn", w_star, x)
    return x, y, w_star


def build_fixed_train_dataset(m, n, d, device):
    x, y, w_star = sample_task_batch(m, n=n, d=d, device=device)
    return {"X": x, "y": y, "w_star": w_star, "M": m, "N": n}


def sample_train_minibatch(train_data, batch_size, device):
    idx = torch.randint(0, train_data["M"], (batch_size,), device=device)
    return train_data["X"][idx], train_data["y"][idx], train_data["w_star"][idx]


def sample_w0_random(x, d, device):
    return torch.randn(x.shape[0], d, device=device)


def build_z(x, y, w_prefix, de, d):
    b, _, n = x.shape
    i = len(w_prefix) - 1
    t = n + i + 1
    z = torch.zeros(b, de, t, device=x.device)
    z[:, :d, :n] = x
    z[:, d, :n] = y
    for step, w_t in enumerate(w_prefix):
        z[:, d + 1 : d + 1 + d, n + step] = w_t
    z[:, -1, -1] = 1.0
    return z


def build_attn_mask(t, n, device, mode="data_only"):
    mask = torch.zeros(t, t, device=device)
    if mode == "data_only":
        mask[-1, n:] = float("-inf")
    elif mode == "history_cot":
        mask[-1, t - 1] = float("-inf")
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return mask


def f_softmax_last(z, v, w, n, mask_mode="data_only"):
    _, de, t = z.shape
    z_last = z[:, :, -1]
    wz_last = torch.einsum("ij,bj->bi", w, z_last)
    logits = torch.einsum("bdt,bd->bt", z, wz_last) / np.sqrt(de)
    attn_mask = build_attn_mask(t=t, n=n, device=z.device, mode=mask_mode)
    logits = logits + attn_mask[-1].unsqueeze(0)
    attn = torch.softmax(logits, dim=-1)
    weighted = torch.einsum("bdt,bt->bd", z, attn)
    out_last = z_last + torch.einsum("ij,bj->bi", v, weighted)
    return out_last


def no_cot_train_loss(v, w, x, y, w_star, d, de):
    n = x.shape[2]
    w0 = sample_w0_random(x, d, x.device)
    z0 = build_z(x, y, [w0], de=de, d=d)
    pred = f_softmax_last(z0, v, w, n=n, mask_mode="data_only")
    w_hat = pred[:, d + 1 : d + 1 + d]
    return ((w_hat - w_star) ** 2).sum(dim=1).mean()


def parameter_l2_penalty(v, w):
    return v.pow(2).sum() + w.pow(2).sum()


def train_model_no_cot_finite(train_data, d, de, iterations, batch_size, lr, l2_lambda, device):
    v = nn.Parameter(0.01 * torch.randn(de, de, device=device))
    w = nn.Parameter(0.01 * torch.randn(de, de, device=device))
    opt = torch.optim.Adam([v, w], lr=lr)
    for _ in range(iterations):
        x, y, w_star = sample_train_minibatch(train_data, batch_size, device)
        data_loss = no_cot_train_loss(v, w, x, y, w_star, d=d, de=de)
        reg_loss = l2_lambda * parameter_l2_penalty(v, w)
        loss = data_loss + reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return v.detach(), w.detach()


def eval_dynamics_curve(v, w, n, t_list, d, de, batch_size, use_history, mask_mode, device):
    x, y, w_star = sample_task_batch(batch_size, n=n, d=d, device=device)
    with torch.no_grad():
        w_hat = sample_w0_random(x, d, device)
        losses = {0: ((w_hat - w_star) ** 2).mean(dim=1).mean().item()}
        max_t = max(t_list)
        if use_history:
            w_hist = [w_hat]
            for t in range(1, max_t + 1):
                z = build_z(x, y, w_hist, de=de, d=d)
                pred = f_softmax_last(z, v, w, n=n, mask_mode=mask_mode)
                w_hat = pred[:, d + 1 : d + 1 + d]
                losses[t] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()
                w_hist.append(w_hat)
        else:
            for t in range(1, max_t + 1):
                z = build_z(x, y, [w_hat], de=de, d=d)
                pred = f_softmax_last(z, v, w, n=n, mask_mode=mask_mode)
                w_hat = pred[:, d + 1 : d + 1 + d]
                losses[t] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()
    return {t: losses[t] for t in t_list}


def save_single_result(outdir, config, curve):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{key}={value}" for key, value in config.items()) + "\n")
    np.savetxt(
        os.path.join(outdir, "curve.csv"),
        np.array([[t, curve[t]] for t in sorted(curve)], dtype=float),
        delimiter=",",
        fmt="%.10e",
    )


def main():
    parser = argparse.ArgumentParser(description="Run one fixed (D, M, N) softmax experiment.")
    parser.add_argument("--d", type=int, default=50, help="Dimension D.")
    parser.add_argument("--m", type=int, default=1000, help="Number of tasks M.")
    parser.add_argument("--n", type=int, default=100, help="Number of in-context samples N.")
    parser.add_argument("--batch-size", type=int, default=300, help="Train minibatch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Eval batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization weight.")
    parser.add_argument("--iterations", type=int, default=5000, help="Training iterations.")
    parser.add_argument("--seed", type=int, default=941, help="Random seed.")
    parser.add_argument(
        "--mask-mode",
        type=str,
        default=ATTN_MASK_MODE,
        choices=["data_only", "history_cot"],
        help="Attention mask mode at inference.",
    )
    parser.add_argument(
        "--use-history",
        action="store_true",
        help="Use history-based CoT style updates at inference.",
    )
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory.")
    args = parser.parse_args()

    de = 2 * args.d + 2
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = build_fixed_train_dataset(args.m, n=args.n, d=args.d, device=DEVICE)
    v_tr, w_tr = train_model_no_cot_finite(
        train_data,
        d=args.d,
        de=de,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_lambda=args.l2,
        device=DEVICE,
    )
    eval_curve = eval_dynamics_curve(
        v_tr,
        w_tr,
        n=args.n,
        t_list=EVAL_T_LIST,
        d=args.d,
        de=de,
        batch_size=args.eval_batch_size,
        use_history=args.use_history if args.use_history else USE_HISTORY_AT_INFERENCE,
        mask_mode=args.mask_mode,
        device=DEVICE,
    )

    config = {
        "timestamp": datetime.now().isoformat(),
        "model": "softmax_attention",
        "D": args.d,
        "M": args.m,
        "N": args.n,
        "seed": args.seed,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "adam_lr": args.lr,
        "param_l2_lambda": args.l2,
        "eval_t_list": ",".join(map(str, EVAL_T_LIST)),
        "mask_mode": args.mask_mode,
        "use_history_at_inference": args.use_history if args.use_history else USE_HISTORY_AT_INFERENCE,
        "device": DEVICE,
    }
    save_single_result(args.outdir, config, eval_curve)
    print(f"Saved single softmax experiment to: {args.outdir}")


if __name__ == "__main__":
    main()
