import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ==================================
# Learn parameters once at fixed M, N
# and visualize the learned matrices.
# ==================================
D = 10
FIXED_M = 20
FIXED_N = 20

BATCH_SIZE = 300
ADAM_LR = 1e-3
PARAM_L2_LAMBDA = 0.001
ITERATIONS = 5000
SEED = 941

DE = 2 * D + 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_task_batch(batch_size, n, d=D, device=device):
    x = torch.randn(batch_size, d, n, device=device)
    w_star = torch.randn(batch_size, d, device=device)
    y = torch.einsum("bd,bdn->bn", w_star, x)
    return x, y, w_star


def build_fixed_train_dataset(m, n):
    x, y, w_star = sample_task_batch(m, n=n)
    return {"X": x, "y": y, "w_star": w_star, "M": m, "N": n}


def sample_train_minibatch(train_data, batch_size=BATCH_SIZE):
    idx = torch.randint(0, train_data["M"], (batch_size,), device=device)
    return train_data["X"][idx], train_data["y"][idx], train_data["w_star"][idx]


def sample_w0_random(x, d=D, device=device):
    return torch.randn(x.shape[0], d, device=device)


def build_Z(x, y, w_prefix):
    b, d, n = x.shape
    i = len(w_prefix) - 1
    t = n + i + 1
    z = torch.zeros(b, DE, t, device=x.device)
    z[:, :d, :n] = x
    z[:, d, :n] = y
    for step, w_t in enumerate(w_prefix):
        z[:, d + 1 : d + 1 + d, n + step] = w_t
    z[:, -1, -1] = 1.0
    return z


def build_data_src_mask(t, n, device):
    mask = torch.zeros(t, device=device)
    mask[:n] = 1.0
    return mask


def f_lsa_last(z, v, w, n, src_mask=None):
    z_last = z[:, :, -1]
    wz = torch.einsum("ij,bj->bi", w, z_last)
    scores = torch.einsum("bdt,bd->bt", z, wz) / n
    if src_mask is not None:
        if src_mask.dim() == 1:
            src_mask = src_mask.unsqueeze(0)
        scores = scores * src_mask
    weighted = torch.einsum("bdt,bt->bd", z, scores)
    out_last = z_last + torch.einsum("ij,bj->bi", v, weighted)
    return out_last


def no_cot_train_loss(v, w, x, y, w_star):
    n = x.shape[2]
    w0 = sample_w0_random(x, D, x.device)
    z0 = build_Z(x, y, [w0])
    src_mask = build_data_src_mask(z0.shape[2], n=n, device=z0.device)
    pred = f_lsa_last(z0, v, w, n=n, src_mask=src_mask)
    w_hat = pred[:, D + 1 : D + 1 + D]
    return ((w_hat - w_star) ** 2).sum(dim=1).mean()


def parameter_l2_penalty(v, w):
    return v.pow(2).sum() + w.pow(2).sum()


def train_model_no_cot_finite(train_data, iterations=ITERATIONS):
    v = nn.Parameter(0.01 * torch.randn(DE, DE, device=device))
    w = nn.Parameter(0.01 * torch.randn(DE, DE, device=device))
    opt = torch.optim.Adam([v, w], lr=ADAM_LR)

    for _ in range(iterations):
        x, y, w_star = sample_train_minibatch(train_data, BATCH_SIZE)
        data_loss = no_cot_train_loss(v, w, x, y, w_star)
        reg_loss = PARAM_L2_LAMBDA * parameter_l2_penalty(v, w)
        loss = data_loss + reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return -v.detach().cpu(), -w.detach().cpu()


def save_outputs(outdir, v_matrix, w_matrix, m, n):
    os.makedirs(outdir, exist_ok=True)

    cfg_lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"D={D}",
        f"M={m}",
        f"N={n}",
        f"BATCH_SIZE={BATCH_SIZE}",
        f"ADAM_LR={ADAM_LR}",
        f"PARAM_L2_LAMBDA={PARAM_L2_LAMBDA}",
        f"ITERATIONS={ITERATIONS}",
        f"SEED={SEED}",
        f"device={device}",
    ]
    with open(os.path.join(outdir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cfg_lines) + "\n")

    np.savetxt(os.path.join(outdir, "v_matrix.csv"), v_matrix.numpy(), delimiter=",", fmt="%.10e")
    np.savetxt(os.path.join(outdir, "w_matrix.csv"), w_matrix.numpy(), delimiter=",", fmt="%.10e")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax, matrix, title in [
        (axes[0], v_matrix.numpy(), "Learned V"),
        (axes[1], w_matrix.numpy(), "Learned W"),
    ]:
        vmax = float(np.abs(matrix).max())
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title} (shape={matrix.shape[0]}x{matrix.shape[1]})")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(os.path.join(outdir, "learned_parameter_heatmaps.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=FIXED_M)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Sequence length used in training. If omitted, --l or the default value is used.",
    )
    parser.add_argument(
        "--l",
        type=int,
        default=None,
        help="Alias of --n. Included because L is sometimes used for the sequence length.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "latest_learned_parameter_heatmap"),
    )
    args = parser.parse_args()
    n = args.l if args.l is not None else args.n
    if n is None:
        n = FIXED_N
    output_dir = os.path.join(args.outdir, f"M{args.m}_N{n}")

    print(
        f"device={device}, D={D}, M={args.m}, N={n}, "
        f"B={BATCH_SIZE}, iters={ITERATIONS}, lr={ADAM_LR}, l2={PARAM_L2_LAMBDA}"
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    run_seed = SEED + 1000 * int(args.m) + int(n)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    train_data = build_fixed_train_dataset(args.m, n=n)
    v_tr, w_tr = train_model_no_cot_finite(train_data)
    save_outputs(output_dir, v_tr, w_tr, args.m, n)
    print(f"Saved heatmaps and raw matrices to: {output_dir}")


if __name__ == "__main__":
    main()
