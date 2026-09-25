"""Distill a small CPU-deployable student predictor from the teacher.

Loss (standardized target space):
    L = sum_j wts_j * [ alpha * SmoothL1(student_j, teacher_j)
                        + (1-alpha) * SmoothL1(student_j, y_j) ]
Stops when validation 10Acc >= target on all 6 outputs.
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teacher"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SeerNetMulti, SeerNetConfig, count_parameters  # noqa: E402
import pipeline as P  # noqa: E402
from train_teacher import load, collate, evaluate, DEV, CACHE  # noqa: E402


@torch.no_grad()
def teacher_soft(net, all_idx, samples, G, st, bs=1024):
    net.eval()
    out = np.zeros((len(samples), 6), dtype=np.float32)
    for i in range(0, len(all_idx), bs):
        idxs = all_idx[i:i + bs]
        d, _, _ = collate(idxs, samples, G, st)
        p = net(d).cpu().numpy()
        for k, gi in enumerate(idxs):
            out[gi] = p[k]
    return out  # standardized space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, default=str(ROOT / "teacher/teacher_final.pt"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="distillation weight")
    ap.add_argument("--wts", type=str, default="4,0.2,1,4,0.2,1")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--tag", type=str, default="student")
    args = ap.parse_args()
    wts = torch.tensor([float(x) for x in args.wts.split(",")], dtype=torch.float32, device=DEV)

    G, meta, st = load()
    samples = meta["samples"]; tr = meta["train_idx"]; va = meta["val_idx"]

    # teacher
    tck = torch.load(args.teacher, map_location=DEV, weights_only=False)
    teacher = SeerNetMulti(SeerNetConfig(**tck["cfg"])).to(DEV)
    teacher.load_state_dict(tck["model"]); teacher.eval()
    print(f"teacher params={count_parameters(teacher)} loaded from {args.teacher}", flush=True)
    soft = teacher_soft(teacher, list(range(len(samples))), samples, G, st)  # [Nsamp,6] std space
    soft_t = torch.from_numpy(soft).to(DEV)

    # student
    cfg = SeerNetConfig(node_dim=st["node_dim"], edge_dim=st["edge_dim"], global_dim=st["global_dim"],
                        hidden=args.hidden, num_blocks=args.blocks, num_outputs=6, head_hidden=args.hidden,
                        metric_heads="separate", activation="relu", dropout=args.dropout,
                        encoder_norm="layernorm", block_norm="prenorm", residual="gated",
                        use_synmm=True, global_agg="synmm", use_gnpb=True, include_u_in_edge_update=True,
                        mlp_z_num_linear_layers=3)
    net = SeerNetMulti(cfg).to(DEV)
    print(f"student params={count_parameters(net)} hidden={args.hidden} blocks={args.blocks}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8, min_lr=1e-6)
    lossf = nn.SmoothL1Loss(reduction="none")
    rng = np.random.default_rng(0)
    logf = open(ROOT / "record/teacher_runs" / f"{args.tag}.log", "w")
    best_min = -1; best_epoch = -1
    ckpt = CACHE / f"{args.tag}_best.pt"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train(); order = rng.permutation(len(tr)); tot = 0.0; nb = 0
        for i in range(0, len(tr), args.bs):
            idxs = [tr[j] for j in order[i:i + args.bs]]
            d, y_std, _ = collate(idxs, samples, G, st)
            tsoft = soft_t[torch.tensor(idxs, device=DEV)]
            opt.zero_grad()
            out = net(d)
            hard = lossf(out, y_std); dist = lossf(out, tsoft)
            loss = ((args.alpha * dist + (1 - args.alpha) * hard) * wts).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        accs, mapes = evaluate(net, va, samples, G, st)
        mn = float(accs.min()); sched.step(mn)
        line = (f"ep{ep:3d} loss={tot/nb:.4f} min10Acc={mn:.3f} "
                f"acc=[{' '.join(f'{a:.3f}' for a in accs)}] "
                f"mape=[{' '.join(f'{m:.1f}' for m in mapes)}] lr={opt.param_groups[0]['lr']:.1e} t={time.time()-t0:.0f}s")
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
        if mn > best_min:
            best_min = mn; best_epoch = ep
            torch.save({"model": net.state_dict(), "cfg": cfg.to_dict(), "stats": st,
                        "accs": accs.tolist(), "epoch": ep, "targets": P.TARGET_NAMES,
                        "params": count_parameters(net)}, ckpt)
        if mn >= args.target:
            print(f"SUCCESS ep{ep} all6>={args.target}: {[f'{n}={a:.3f}' for n,a in zip(P.TARGET_NAMES,accs)]}", flush=True)
            logf.write("SUCCESS\n"); break
    print(f"DONE best_min10Acc={best_min:.3f} at ep{best_epoch} params={count_parameters(net)} ckpt={ckpt}", flush=True)
    logf.write(f"DONE best_min10Acc={best_min:.3f} at ep{best_epoch}\n"); logf.close()


if __name__ == "__main__":
    main()
