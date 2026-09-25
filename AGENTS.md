# Nautilus Rules

- Follow the user guidance of Nautilus on `https://nrp.ai/documentation/userdocs`

- Apply for Nautilus with a PyTorch CUDA container image

- After submitting any Nautilus Kubernetes job, immediately collect feedback with `kubectl get job --namespace`, `kubectl get pod --namespace`, `kubectl describe pod --namespace`, `kubectl logs --namespace`, and recent events before doing other work

- After launching a Nautilus job, start a local `nohup` monitor loop that records job status, pod status, pod events, process status, persistent training-log tail, and checkpoint/output files at least once every 60 seconds for 1st 5 mins and every 20 min after first 5 min

- Keep the Nautilus monitor log in this repository under `record/` or another tracked project path, and report its path to the user

- Do not treat a background monitor log as user feedback by itself; while the conversation turn is active, report Nautilus job progress back to the user at least once every 60 seconds until the job is stable, failed, or the user asks to stop

- If a Nautilus job enters `Failed`, immediately report the failed pod name, exit code, last relevant log lines, and the next corrective action

- Must, must use a verifier for each of your action

# General Rules

- Be very careful to make sure dataset has variants of model depth, hidden dim, batch size, data type.

- The model needs a feedback system to correct its output during inference based on real usage of its power and VRAM

- The model needs to be small and fast enough to be deployed on a intel server CPU

- You can now use either `ssh ABA` (When at least 1 GPU on this server is available) or request at most 2 A100s on Nautilus server to train our predictor

- When given accuracy, give its accuracy on validation/test set against GT within 5% and 10% error seperately
