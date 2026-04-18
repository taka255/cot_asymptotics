import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


# ==================================
# Simple finite-M dynamics experiment
# (fixed D; vary M, N)
# softmax attention + data-only/history mask
# ==================================
D = 50
#M_LIST = [10, 100, 1000, 10000]
#N_LIST = [10, 100, 1000]

# log10 空間で等間隔に 15 点取り、最終的に整数化
M_LIST = np.round(np.logspace(2, 4, 15)).astype(int).tolist()
N_LIST = np.round(np.logspace(1, 3, 15)).astype(int).tolist()

BATCH_SIZE = 300
ADAM_LR = 1e-3
PARAM_L2_LAMBDA = 0.001
ITERATIONS = 5000
EVAL_T_LIST = list(range(0, 32))
T_HEATMAP = 28
EVAL_BATCH_SIZE = 4096
SEED = 941
N_TRIALS = 1  # fixed by design

USE_HISTORY_AT_INFERENCE = True
ATTN_MASK_MODE = "data_only"  # "data_only" | "history_cot"

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
    #! last token is 1.0 (not : in all previous cases)
    z[:, -1, -1] = 1.0
    return z


def build_attn_mask(T, L, device, mode="data_only"):
    # user-specified base style
    attn_mask = torch.zeros(T, T, device=device)

    if mode == "data_only":
        attn_mask[-1, L:] = float("-inf")
    elif mode == "history_cot":
        # allow examples + past estimate tokens, block only self
        attn_mask[-1, T - 1] = float("-inf")
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return attn_mask


def f_softmax_last(z, v, w, n, mask_mode="data_only"):
    b, de, t = z.shape
    z_last = z[:, :, -1]
    wz_last = torch.einsum("ij,bj->bi", w, z_last)
    logits = torch.einsum("bdt,bd->bt", z, wz_last) / np.sqrt(de)
    attn_mask = build_attn_mask(T=t, L=n, device=z.device, mode=mask_mode)
    logits = logits + attn_mask[-1].unsqueeze(0)
    attn = torch.softmax(logits, dim=-1)
    weighted = torch.einsum("bdt,bt->bd", z, attn)
    out_last = z_last + torch.einsum("ij,bj->bi", v, weighted)
    return out_last


def no_cot_train_loss(v, w, x, y, w_star):
    n = x.shape[2]
    w0 = sample_w0_random(x, D, x.device)
    z0 = build_Z(x, y, [w0])
    pred = f_softmax_last(z0, v, w, n=n, mask_mode="data_only")
    w_hat = pred[:, D + 1 : D + 1 + D]
    return ((w_hat - w_star) ** 2).sum(dim=1).mean()


def eval_dynamics_curve(
    v,
    w,
    n,
    t_list,
    batch_size=EVAL_BATCH_SIZE,
    use_history=USE_HISTORY_AT_INFERENCE,
    mask_mode=ATTN_MASK_MODE,
):
    x, y, w_star = sample_task_batch(batch_size, n=n)
    with torch.no_grad():
        w_hat = sample_w0_random(x, D, device)
        losses = {}
        losses[0] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()
        max_t = max(t_list)

        if use_history:
            w_hist = [w_hat]
            for t in range(1, max_t + 1):
                z = build_Z(x, y, w_hist)
                pred = f_softmax_last(z, v, w, n=n, mask_mode=mask_mode)
                w_hat = pred[:, D + 1 : D + 1 + D]
                losses[t] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()
                w_hist.append(w_hat)
        else:
            for t in range(1, max_t + 1):
                z = build_Z(x, y, [w_hat])
                pred = f_softmax_last(z, v, w, n=n, mask_mode=mask_mode)
                w_hat = pred[:, D + 1 : D + 1 + D]
                losses[t] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()

    return {t: losses[t] for t in t_list}


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

    return v.detach(), w.detach()


def save_raw_results(outdir, results_by_m_n):
    os.makedirs(outdir, exist_ok=True)

    # config
    cfg_lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"D={D}",
        f"M_LIST={','.join(map(str, M_LIST))}",
        f"N_LIST={','.join(map(str, N_LIST))}",
        f"BATCH_SIZE={BATCH_SIZE}",
        f"ADAM_LR={ADAM_LR}",
        f"PARAM_L2_LAMBDA={PARAM_L2_LAMBDA}",
        f"ITERATIONS={ITERATIONS}",
        f"EVAL_T_LIST={','.join(map(str, EVAL_T_LIST))}",
        f"T_HEATMAP={T_HEATMAP}",
        f"EVAL_BATCH_SIZE={EVAL_BATCH_SIZE}",
        f"SEED={SEED}",
        f"N_TRIALS={N_TRIALS}",
        f"USE_HISTORY_AT_INFERENCE={USE_HISTORY_AT_INFERENCE}",
        f"ATTN_MASK_MODE={ATTN_MASK_MODE}",
        f"device={device}",
    ]
    with open(os.path.join(outdir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cfg_lines) + "\n")

    # lists
    np.savetxt(os.path.join(outdir, "M_list.csv"), np.array(M_LIST, dtype=int), delimiter=",", fmt="%d")
    np.savetxt(os.path.join(outdir, "N_list.csv"), np.array(N_LIST, dtype=int), delimiter=",", fmt="%d")
    np.savetxt(
        os.path.join(outdir, "EVAL_T_list.csv"),
        np.array(EVAL_T_LIST, dtype=int),
        delimiter=",",
        fmt="%d",
    )

    # long raw data: m, n, t, error
    rows = []
    for m in M_LIST:
        for n in N_LIST:
            for t in EVAL_T_LIST:
                rows.append([m, n, t, results_by_m_n[m][n]["mean"][t]])
    long_arr = np.array(rows, dtype=float)
    np.savetxt(os.path.join(outdir, "curves_long.csv"), long_arr, delimiter=",", fmt="%.10e")

    # heatmap at fixed t
    heat = np.zeros((len(M_LIST), len(N_LIST)), dtype=float)
    for i, m in enumerate(M_LIST):
        for j, n in enumerate(N_LIST):
            heat[i, j] = results_by_m_n[m][n]["mean"][T_HEATMAP]
    np.savetxt(os.path.join(outdir, f"heat_t{T_HEATMAP}.csv"), heat, delimiter=",", fmt="%.10e")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "latest_softmax_heatmap"),
    )
    args = parser.parse_args()

    print(
        f"device={device}, D={D}, M_LIST={M_LIST}, N_LIST={N_LIST}, "
        f"B={BATCH_SIZE}, iters={ITERATIONS}, lr={ADAM_LR}, l2={PARAM_L2_LAMBDA}, "
        f"trials={N_TRIALS}"
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    results_by_m_n = {}
    for m in M_LIST:
        results_by_m_n[m] = {}
        for n in N_LIST:
            run_seed = SEED + 100000 * int(m) + int(n)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            train_data = build_fixed_train_dataset(m, n=n)
            v_tr, w_tr = train_model_no_cot_finite(train_data)
            eval_curve = eval_dynamics_curve(v_tr, w_tr, n=n, t_list=EVAL_T_LIST)

            results_by_m_n[m][n] = {
                "mean": {t: float(eval_curve[t]) for t in EVAL_T_LIST},
                "se": {t: 0.0 for t in EVAL_T_LIST},
            }
            print(f"M={m}, N={n}: done")

    save_raw_results(args.outdir, results_by_m_n)
    print(f"Saved raw outputs to: {args.outdir}")


if __name__ == "__main__":
    main()

