import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import json
from datetime import datetime
from pathlib import Path


# =========================
# Simple finite-M experiment
# (fixed D, N; vary M)
# =========================
D = 50

NM_LIST = [
    (30, 200),
    (10, 10_000),
    (30, 10_000),
    (100, 10_000),
    (200, 10_000),
    (400, 10_000),
    (600, 10_000),
    (800, 10_000),
]
BATCH_SIZE = 500
ADAM_LR = 1e-3
PARAM_L2_LAMBDA = 0.0001
ITERATIONS = 3000
EVAL_T_LIST = list(range(0, 22))
EVAL_BATCH_SIZE = 2048
SEED = 120
N_TRIALS = 5  # 同一設定での繰り返し回数（SE算出用）

DE = 2 * D + 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"device={device}, "
    f"B={BATCH_SIZE}, iters={ITERATIONS}, lr={ADAM_LR}, l2={PARAM_L2_LAMBDA}, "
    f"trials={N_TRIALS}"
)


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
    # Initial estimate is random normal (requested setting).
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
    return 0.5 * ((w_hat - w_star) ** 2).sum(dim=1).mean()


def eval_dynamics_curve(v, w, n, t_list, batch_size=EVAL_BATCH_SIZE):
    x, y, w_star = sample_task_batch(batch_size, n=n)
    with torch.no_grad():
        w_hat = sample_w0_random(x, D, device)
        losses = {}
        losses[0] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()
        max_t = max(t_list)
        for t in range(1, max_t + 1):
            z = build_Z(x, y, [w_hat])
            src_mask = build_data_src_mask(z.shape[2], n=n, device=z.device)
            pred = f_lsa_last(z, v, w, n=n, src_mask=src_mask)
            w_hat = pred[:, D + 1 : D + 1 + D]
            losses[t] = ((w_hat - w_star) ** 2).mean(dim=1).mean().item()
    return {t: losses[t] for t in t_list}


def parameter_l2_penalty(v, w):
    return v.pow(2).sum() + w.pow(2).sum()


def train_model_no_cot_finite(train_data, iterations=ITERATIONS):
    v = nn.Parameter(0.01 * torch.randn(DE, DE, device=device))
    w = nn.Parameter(0.01 * torch.randn(DE, DE, device=device))
    opt = torch.optim.Adam([v, w], lr=ADAM_LR)
    loss_history = []

    for _ in range(iterations):
        x, y, w_star = sample_train_minibatch(train_data, BATCH_SIZE)
        data_loss = no_cot_train_loss(v, w, x, y, w_star)
        reg_loss = PARAM_L2_LAMBDA * parameter_l2_penalty(v, w)
        loss = data_loss + reg_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().item()))

    return v.detach(), w.detach(), np.array(loss_history, dtype=np.float32)

np.random.seed(SEED)
torch.manual_seed(SEED)

results_by_nm = {}
trial_metadata = []
raw_eval_curves_to_save = {}
train_loss_histories_to_save = {}
for n, m in NM_LIST:
    trial_curves = []

    for trial in range(N_TRIALS):
        # (N, M)ごとに同じ乱数系列を使わないように試行seedを分離
        trial_seed = SEED + 1000 * trial + 17 * n + m
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        train_data = build_fixed_train_dataset(m, n)
        v_tr, w_tr, train_loss_history = train_model_no_cot_finite(train_data)
        eval_curve = eval_dynamics_curve(v_tr, w_tr, n=n, t_list=EVAL_T_LIST)
        trial_curves.append(np.array([eval_curve[t] for t in EVAL_T_LIST]))
        train_loss_histories_to_save[f"n{n}_m{m}_trial{trial}"] = train_loss_history
        trial_metadata.append(
            {
                "n": int(n),
                "m": int(m),
                "trial": int(trial),
                "trial_seed": int(trial_seed),
                "eval_t_list": [int(t) for t in EVAL_T_LIST],
            }
        )

    trial_curves = np.stack(trial_curves, axis=0)  # [trials, len(t)]
    mean_curve = trial_curves.mean(axis=0)
    se_curve = trial_curves.std(axis=0, ddof=1) / np.sqrt(N_TRIALS)
    raw_eval_curves_to_save[f"n{n}_m{m}"] = trial_curves.astype(np.float32)

    results_by_nm[(n, m)] = {
        "mean": {t: float(mean_curve[i]) for i, t in enumerate(EVAL_T_LIST)},
        "se": {t: float(se_curve[i]) for i, t in enumerate(EVAL_T_LIST)},
        "all": trial_curves,
    }

    print(f"N={n}, M={m}: done ({N_TRIALS} trials)")




# data save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(__file__).resolve().parent / "results" / f"linear_dynamics_{run_id}"
save_dir.mkdir(parents=True, exist_ok=True)

config = {
    "run_id": run_id,
    "device": str(device),
    "D": int(D),
    "DE": int(DE),
    "NM_LIST": [[int(n), int(m)] for n, m in NM_LIST],
    "BATCH_SIZE": int(BATCH_SIZE),
    "ADAM_LR": float(ADAM_LR),
    "PARAM_L2_LAMBDA": float(PARAM_L2_LAMBDA),
    "ITERATIONS": int(ITERATIONS),
    "EVAL_T_LIST": [int(t) for t in EVAL_T_LIST],
    "EVAL_BATCH_SIZE": int(EVAL_BATCH_SIZE),
    "SEED": int(SEED),
    "N_TRIALS": int(N_TRIALS),
}
with (save_dir / "config.json").open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

with (save_dir / "trial_meta.json").open("w", encoding="utf-8") as f:
    json.dump(trial_metadata, f, indent=2)

summary_csv_path = save_dir / "summary.csv"
with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "m", "t", "mean_loss", "se_loss"])
    for n, m in NM_LIST:
        for t in EVAL_T_LIST:
            writer.writerow(
                [
                    int(n),
                    int(m),
                    int(t),
                    float(results_by_nm[(n, m)]["mean"][t]),
                    float(results_by_nm[(n, m)]["se"][t]),
                ]
            )

np.savez_compressed(save_dir / "raw_eval_curves.npz", **raw_eval_curves_to_save)
np.savez_compressed(save_dir / "train_loss_histories.npz", **train_loss_histories_to_save)

print(f"Saved all run artifacts to: {save_dir}")
