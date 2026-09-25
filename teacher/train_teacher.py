"""Train the SeerNetMulti teacher predictor to >=80% 10Acc on all 6 targets."""
from __future__ import annotations
import argparse, os, pickle, time, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predictor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SeerNetMulti, SeerNetConfig, count_parameters  # noqa: E402
from metrics import x_acc, mape  # noqa: E402
import pipeline as P  # noqa: E402

CACHE = Path(os.environ.get("SEER_CACHE", Path(__file__).resolve().parent / "cache"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Bunch:
    pass


def load():
    graphs = pickle.load(open(CACHE / "graphs.pkl", "rb"))
    meta = pickle.load(open(CACHE / "meta.pkl", "rb"))
    st = meta["stats"]
    xm, xs = st["x_mean"], st["x_std"]; em, es = st["e_mean"], st["e_std"]; gm, gs = st["g_mean"], st["g_std"]
    G = {}
    for mid, (xo, xc, ei, ec, gc) in graphs.items():
        x = np.concatenate([xo, (xc - xm) / xs], axis=1).astype(np.float32)
        e = ((ec - em) / es).astype(np.float32) if ec.shape[0] else np.zeros((0, 3), np.float32)
        g = ((gc - gm) / gs).astype(np.float32)
        G[mid] = (torch.from_numpy(x), torch.from_numpy(ei), torch.from_numpy(e), torch.from_numpy(g))
    return G, meta, st


def collate(idxs, samples, G, st):
    xs, eis, es, us, ys, batch = [], [], [], [], [], []
    off = 0
    for bi, k in enumerate(idxs):
        mid, prec, y6 = samples[k]
        x, ei, e, g = G[mid]
        xs.append(x); es.append(e)
        eis.append(ei + off)
        prec_oh = torch.zeros(len(P.PRECISIONS), dtype=torch.float32)
        prec_oh[P.PREC_INDEX[prec]] = 1.0
        us.append(torch.cat([g, prec_oh]))
        batch.append(torch.full((x.shape[0],), bi, dtype=torch.long))
        ys.append(y6)
        off += x.shape[0]
    d = Bunch()
    d.x = torch.cat(xs).to(DEV)
    d.edge_index = torch.cat(eis, dim=1).to(DEV)
    d.edge_attr = torch.cat(es).to(DEV) if any(e.shape[0] for e in es) else torch.zeros((0, 3), device=DEV)
    d.u = torch.stack(us).to(DEV)
    d.batch = torch.cat(batch).to(DEV)
    d.num_graphs = len(idxs)
    y_raw = np.stack(ys)
    ylog = np.log1p(np.maximum(y_raw, 0.0))
    y_std = torch.from_numpy(((ylog - st["y_mean"]) / st["y_std"]).astype(np.float32)).to(DEV)
    return d, y_std, y_raw


def invert(pred_std, st):
    ylog = pred_std * st["y_std"] + st["y_mean"]
    return np.maximum(np.expm1(ylog), 0.0)


@torch.no_grad()
def evaluate(net, idxs, samples, G, st, bs=1024):
    net.eval()
    preds, trues = [], []
    for i in range(0, len(idxs), bs):
        d, _, y_raw = collate(idxs[i:i + bs], samples, G, st)
        p = net(d).cpu().numpy()
        preds.append(p); trues.append(y_raw)
    pred_std = np.concatenate(preds); y_raw = np.concatenate(trues)
    pred_raw = invert(pred_std, st)
    accs, mapes = [], []
    for j in range(6):
        accs.append(x_acc(y_raw[:, j], pred_raw[:, j], 10.0))
        mapes.append(mape(y_raw[:, j], pred_raw[:, j]))
    return np.array(accs), np.array(mapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--tag", type=str, default="teacher")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--wts", type=str, default="1,1,1,1,1,1",
                    help="per-target loss weights: train_util,train_mem,train_time,infer_util,infer_mem,infer_time")
    args = ap.parse_args()
    wts = torch.tensor([float(x) for x in args.wts.split(",")], dtype=torch.float32, device=DEV)

    G, meta, st = load()
    samples = meta["samples"]; tr = meta["train_idx"]; va = meta["val_idx"]
    cfg = SeerNetConfig(node_dim=st["node_dim"], edge_dim=st["edge_dim"], global_dim=st["global_dim"],
                        hidden=args.hidden, num_blocks=args.blocks, num_outputs=6, head_hidden=args.hidden,
                        metric_heads="separate", activation="relu", dropout=args.dropout,
                        encoder_norm="layernorm", block_norm="prenorm", residual="gated",
                        use_synmm=True, global_agg="synmm", use_gnpb=True, include_u_in_edge_update=True,
                        mlp_z_num_linear_layers=3)
    net = SeerNetMulti(cfg).to(DEV)
    print(f"device={DEV} params={count_parameters(net)} node={cfg.node_dim} edge={cfg.edge_dim} global={cfg.global_dim}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8, min_lr=1e-6)
    lossf = nn.SmoothL1Loss(reduction="none")
    rng = np.random.default_rng(0)
    logp = ROOT / "record/teacher_runs" / f"{args.tag}.log"
    logf = open(logp, "w")
    best_min = -1; best_epoch = -1
    ckpt = CACHE / f"{args.tag}_best.pt"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(tr))
        tot = 0.0; nb = 0
        for i in range(0, len(tr), args.bs):
            idxs = [tr[j] for j in order[i:i + args.bs]]
            d, y_std, _ = collate(idxs, samples, G, st)
            opt.zero_grad()
            out = net(d)
            loss = (lossf(out, y_std) * wts).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
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
                        "accs": accs.tolist(), "epoch": ep, "targets": P.TARGET_NAMES}, ckpt)
        if mn >= args.target:
            done = (f"SUCCESS ep{ep} all6>= {args.target}: {[f'{n}={a:.3f}' for n,a in zip(P.TARGET_NAMES,accs)]}")
            print(done, flush=True); logf.write(done + "\n"); logf.flush()
            break
    summary = f"DONE best_min10Acc={best_min:.3f} at ep{best_epoch} ckpt={ckpt}"
    print(summary, flush=True); logf.write(summary + "\n"); logf.close()


if __name__ == "__main__":
    main()
