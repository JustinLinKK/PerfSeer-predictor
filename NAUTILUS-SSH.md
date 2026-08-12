# `ssh Nautilus` — How It Works

- One command: `ssh Nautilus`. Everything else is automatic.

## Chain

```
ssh Nautilus
  └─ ~/.ssh/config: Host Nautilus → localhost:2222, user downeyflyfan, key ~/.ssh/id_ed25519
       └─ ProxyCommand ~/.local/bin/nautilus-connect localhost 2222
            1. kubectl get deploy gpu-dev -n ecepxie   → if replicas==0, scale to 1
            2. wait ≤30 min for a Running pod          (GPU queue)
            3. wait for "keep-alive" in pod logs       (init.sh finished, sshd up)
            4. if 2222 not listening → nautilus-pf --daemon (kubectl port-forward 2222:22)
            5. exec nc localhost 2222                  → raw TCP handed to ssh
```

- `StrictHostKeyChecking no` plus `UserKnownHostsFile /dev/null`: the pod host key changes on every restart, so verification is disabled by design.

- `ProxyCommand` uses `~/...` (not an absolute path) because `.ssh/config` is unison-synced Mac↔Omen and the home paths differ; ssh runs ProxyCommand through a shell, which expands `~`.

## Cluster Facts

| Item | Value |
|---|---|
| kube context / namespace | `nautilus` / `ecepxie` |
| deployment | `gpu-dev` (strategy: Recreate) |
| image | `nvcr.io/nvidia/pytorch:24.05-py3` (CUDA 12, so V100 sm 7.0 works) |
| resources | 4 GPU, cpu 8/16, mem 32/64 Gi, `/dev/shm` 16 Gi tmpfs |
| current GPU pin | `Tesla-V100-SXM2-32GB` (required nodeAffinity) |
| persistent home | PVC `yuw-home` mounted at `/root` (100 G cephfs; login lands in `/root/downeyflyfan`) |
| bootstrap | ConfigMap `nautilus-init` mounted at `/init/init.sh`, runs on every pod start |

- `init.sh` apt-installs zsh, sshd, ripgrep, fd, neovim on each boot (ephemeral rootfs), installs starship, yazi and zinit into `/root/.local/bin` once (PVC-persistent), injects the Omen ed25519 public key into `authorized_keys`, starts sshd, then prints `keep-alive`.

## Helper Commands

```bash
nautilus-pf --daemon      # start self-healing port-forward 2222->22 (auto-reconnects on pod restart)
nautilus-pf stop          # kill daemon + port-forward
nautilus-gpu              # fzf-pick GPU type (Ampere+ by default); -p = preferred not required; -v = allow V100
nautilus-scale N          # jupyter-a10 deployment: N GPUs, cpu 2N/4N, mem 8N/16N Gi
nautilus-a10-watch        # poll for free A10 capacity; --switch repoints gpu-dev when it appears
```

- `nautilus-gpu` without `-p` makes the GPU **required**, so the pod queues as Pending if none is free and `ssh Nautilus` blocks waiting. With `-p` it falls back to any other GPU above the compute-capability floor.

## Diagnose

```bash
kubectl get pods -n ecepxie -l app=gpu-dev
kubectl logs -n ecepxie -l app=gpu-dev --tail=50
ss -tln | grep 2222
cat /tmp/nautilus-pf.log
```

| Symptom | Cause / fix |
|---|---|
| hangs at "waiting for a Running pod" | GPU queued; wait, or run `nautilus-gpu -p` to allow fallback |
| "Connection refused" | port-forward points at an old/terminating pod; run `nautilus-pf --daemon` |
| hangs after the pod is Running | `init.sh` still installing; wait for `keep-alive` in the logs |
| `kubectl not found` | ProxyCommand aborts immediately; fix `PATH` |

## Notes

- The deployment stays at 1 replica. To free GPUs: `kubectl scale deploy gpu-dev -n ecepxie --replicas=0`. The next `ssh Nautilus` scales it back up.

- Only `/root` (the PVC) survives a pod restart. Everything outside it is lost.

- A pod restart kills all running jobs, so use `tmux` or `nohup` inside the pod.
