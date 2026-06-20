# Nautilus E2E verify2c-20260619184849

- Namespace: ecepxie

- PVC: test-pvc

- GPUs: a40, a100


$ kubectl get jobs -n ecepxie -l perfseer-run-id=verify2c-20260619184849 -o wide
NAME                                        STATUS    COMPLETIONS   DURATION   AGE   CONTAINERS      IMAGES                                        SELECTOR
perfseer-e2e-verify2c-20260619184849-a100   Running   0/1           1s         1s    label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=46067c50-0e9a-4fea-a77c-bb8930dc83af
perfseer-e2e-verify2c-20260619184849-a40    Running   0/1           2s         2s    label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=f955ce11-caea-489d-b8a7-824bc05c5646


$ kubectl get pods -n ecepxie -l perfseer-run-id=verify2c-20260619184849 -o wide
NAME                                              READY   STATUS              RESTARTS   AGE   IP       NODE                NOMINATED NODE   READINESS GATES
perfseer-e2e-verify2c-20260619184849-a100-cv8gv   0/1     ContainerCreating   0          1s    <none>   gp-argo.usd.edu     <none>           <none>
perfseer-e2e-verify2c-20260619184849-a40-l8bmt    0/1     ContainerCreating   0          2s    <none>   bak-hpc1.csub.edu   <none>           <none>


$ kubectl get events -n ecepxie --sort-by=.lastTimestamp
LAST SEEN   TYPE      REASON                     OBJECT                                                MESSAGE
14m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4-20260619183403-a10           Created pod: perfseer-e2e-verify4-20260619183403-a10-9mmnt
14m         Normal    Scheduled                  pod/perfseer-e2e-verify4-20260619183403-a40-cxjrv     Successfully assigned ecepxie/perfseer-e2e-verify4-20260619183403-a40-cxjrv to bak-hpc1.csub.edu
14m         Normal    Scheduled                  pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    Successfully assigned ecepxie/perfseer-e2e-verify4-20260619183403-a100-pn8zz to gp-argo.usd.edu
14m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4-20260619183403-a40           Created pod: perfseer-e2e-verify4-20260619183403-a40-cxjrv
14m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4-20260619183403-a100          Created pod: perfseer-e2e-verify4-20260619183403-a100-pn8zz
14m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
14m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4-20260619183403-v100          Created pod: perfseer-e2e-verify4-20260619183403-v100-rd9pb
14m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4-20260619183403-a40-cxjrv     AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
14m         Warning   FailedScheduling           pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    0/524 nodes are available: 1 Insufficient memory, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint { ...
14m         Normal    Scheduled                  pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    Successfully assigned ecepxie/perfseer-e2e-verify4-20260619183403-v100-rd9pb to chi-dgx-node04.csuchico.edu
14m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
14m         Normal    Pulled                     pod/perfseer-e2e-verify4-20260619183403-a40-cxjrv     Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
14m         Normal    Created                    pod/perfseer-e2e-verify4-20260619183403-a40-cxjrv     Created container: label-sampler
14m         Normal    Started                    pod/perfseer-e2e-verify4-20260619183403-a40-cxjrv     Started container label-sampler
14m         Normal    Pulling                    pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    Pulling image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel"
14m         Warning   BackoffLimitExceeded       job/perfseer-e2e-verify4-20260619183403-a40           Job has reached the specified backoff limit
14m         Normal    Pulling                    pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    Pulling image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel"
14m         Warning   FailedScheduling           pod/perfseer-e2e-verify4-20260619183403-a10-9mmnt     0/524 nodes are available: 1 Insufficient memory, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint { ...
12m         Normal    Scheduled                  pod/perfseer-e2e-verify4b-20260619183604-a40-ds6dn    Successfully assigned ecepxie/perfseer-e2e-verify4b-20260619183604-a40-ds6dn to bak-hpc1.csub.edu
12m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4b-20260619183604-a10          Created pod: perfseer-e2e-verify4b-20260619183604-a10-zkj7q
12m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4b-20260619183604-a40          Created pod: perfseer-e2e-verify4b-20260619183604-a40-ds6dn
12m         Normal    Scheduled                  pod/perfseer-e2e-verify4b-20260619183604-a100-tlqrk   Successfully assigned ecepxie/perfseer-e2e-verify4b-20260619183604-a100-tlqrk to nautilus-it-gpu07.fullerton.edu
12m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4b-20260619183604-a100         Created pod: perfseer-e2e-verify4b-20260619183604-a100-tlqrk
12m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4b-20260619183604-a40-ds6dn    AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
12m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4b-20260619183604-v100         Created pod: perfseer-e2e-verify4b-20260619183604-v100-5tb29
12m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4b-20260619183604-a100-tlqrk   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
12m         Warning   FailedScheduling           pod/perfseer-e2e-verify4b-20260619183604-v100-5tb29   0/524 nodes are available: 1 Insufficient memory, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint { ...
12m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4b-20260619183604-v100-5tb29   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
12m         Normal    Pulled                     pod/perfseer-e2e-verify4b-20260619183604-a100-tlqrk   Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
12m         Normal    Created                    pod/perfseer-e2e-verify4b-20260619183604-a100-tlqrk   Created container: label-sampler
12m         Normal    Started                    pod/perfseer-e2e-verify4b-20260619183604-a100-tlqrk   Started container label-sampler
12m         Normal    Pulled                     pod/perfseer-e2e-verify4b-20260619183604-a40-ds6dn    Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
12m         Normal    Created                    pod/perfseer-e2e-verify4b-20260619183604-a40-ds6dn    Created container: label-sampler
12m         Normal    Started                    pod/perfseer-e2e-verify4b-20260619183604-a40-ds6dn    Started container label-sampler
12m         Normal    Pulled                     pod/perfseer-e2e-verify4b-20260619183604-v100-5tb29   Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
12m         Normal    Created                    pod/perfseer-e2e-verify4b-20260619183604-v100-5tb29   Created container: label-sampler
12m         Normal    Started                    pod/perfseer-e2e-verify4b-20260619183604-v100-5tb29   Started container label-sampler
12m         Warning   BackoffLimitExceeded       job/perfseer-e2e-verify4b-20260619183604-a100         Job has reached the specified backoff limit
12m         Normal    Completed                  job/perfseer-e2e-verify4b-20260619183604-a40          Job completed
12m         Normal    Completed                  job/perfseer-e2e-verify4b-20260619183604-v100         Job completed
11m         Warning   FailedScheduling           pod/perfseer-e2e-verify4b-20260619183604-a10-zkj7q    0/524 nodes are available: 1 Insufficient memory, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint { ...
11m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4c-20260619183732-a10          Created pod: perfseer-e2e-verify4c-20260619183732-a10-t688v
11m         Normal    Scheduled                  pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   Successfully assigned ecepxie/perfseer-e2e-verify4c-20260619183732-a100-66mgn to rci-nrp-gpu-04.sdsu.edu
11m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4c-20260619183732-v100         Created pod: perfseer-e2e-verify4c-20260619183732-v100-t2x9r
11m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4c-20260619183732-a100         Created pod: perfseer-e2e-verify4c-20260619183732-a100-66mgn
11m         Normal    Scheduled                  pod/perfseer-e2e-verify4c-20260619183732-a40-4qz49    Successfully assigned ecepxie/perfseer-e2e-verify4c-20260619183732-a40-4qz49 to bak-hpc1.csub.edu
11m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4c-20260619183732-a40-4qz49    AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
11m         Normal    SuccessfulCreate           job/perfseer-e2e-verify4c-20260619183732-a40          Created pod: perfseer-e2e-verify4c-20260619183732-a40-4qz49
11m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
11m         Warning   FailedScheduling           pod/perfseer-e2e-verify4c-20260619183732-v100-t2x9r   0/524 nodes are available: 1 Insufficient memory, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint { ...
11m         Normal    Scheduled                  pod/perfseer-e2e-verify4c-20260619183732-v100-t2x9r   Successfully assigned ecepxie/perfseer-e2e-verify4c-20260619183732-v100-t2x9r to gpn-fiona-mizzou-7.rnet.missouri.edu
11m         Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify4c-20260619183732-v100-t2x9r   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
11m         Normal    Pulling                    pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   Pulling image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel"
11m         Normal    Pulled                     pod/perfseer-e2e-verify4c-20260619183732-a40-4qz49    Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
11m         Normal    Created                    pod/perfseer-e2e-verify4c-20260619183732-a40-4qz49    Created container: label-sampler
11m         Normal    Started                    pod/perfseer-e2e-verify4c-20260619183732-a40-4qz49    Started container label-sampler
11m         Normal    Pulling                    pod/perfseer-e2e-verify4c-20260619183732-v100-t2x9r   Pulling image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel"
11m         Normal    Completed                  job/perfseer-e2e-verify4c-20260619183732-a40          Job completed
10m         Warning   FailedScheduling           pod/perfseer-e2e-verify4c-20260619183732-a10-t688v    0/524 nodes are available: 1 Insufficient memory, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint { ...
9m37s       Normal    Created                    pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    Created container: label-sampler
9m37s       Normal    Pulled                     pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    Successfully pulled image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" in 4m57.587s (4m57.587s including waiting). Image size: 9371344010 bytes.
9m36s       Normal    Killing                    pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    Stopping container label-sampler
9m36s       Normal    Started                    pod/perfseer-e2e-verify4-20260619183403-v100-rd9pb    Started container label-sampler
9m33s       Normal    Created                    pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    Created container: label-sampler
9m33s       Normal    Pulled                     pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    Successfully pulled image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" in 5m5.802s (5m5.802s including waiting). Image size: 9371344010 bytes.
9m32s       Normal    Started                    pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    Started container label-sampler
9m32s       Normal    Killing                    pod/perfseer-e2e-verify4-20260619183403-a100-pn8zz    Stopping container label-sampler
7m38s       Normal    Started                    pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   Started container label-sampler
7m38s       Normal    Created                    pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   Created container: label-sampler
7m38s       Normal    Killing                    pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   Stopping container label-sampler
7m38s       Normal    Pulled                     pod/perfseer-e2e-verify4c-20260619183732-a100-66mgn   Successfully pulled image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" in 3m29.655s (3m29.655s including waiting). Image size: 9371344010 bytes.
6m55s       Normal    SuccessfulCreate           job/perfseer-e2e-verify2-20260619184156-a40           Created pod: perfseer-e2e-verify2-20260619184156-a40-m8sfx
6m55s       Normal    Scheduled                  pod/perfseer-e2e-verify2-20260619184156-a40-m8sfx     Successfully assigned ecepxie/perfseer-e2e-verify2-20260619184156-a40-m8sfx to bak-hpc1.csub.edu
6m55s       Normal    SuccessfulCreate           job/perfseer-e2e-verify2-20260619184156-v100          Created pod: perfseer-e2e-verify2-20260619184156-v100-hd9cs
6m55s       Warning   FailedScheduling           pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    0/524 nodes are available: 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint {nautilus.io/reservation ...
6m54s       Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify2-20260619184156-a40-m8sfx     AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
6m52s       Normal    Created                    pod/perfseer-e2e-verify2-20260619184156-a40-m8sfx     Created container: label-sampler
6m52s       Normal    Pulled                     pod/perfseer-e2e-verify2-20260619184156-a40-m8sfx     Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
6m52s       Normal    Started                    pod/perfseer-e2e-verify2-20260619184156-a40-m8sfx     Started container label-sampler
6m52s       Normal    Scheduled                  pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    Successfully assigned ecepxie/perfseer-e2e-verify2-20260619184156-v100-hd9cs to chi-dgx-node02.csuchico.edu
6m51s       Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
6m46s       Normal    Completed                  job/perfseer-e2e-verify2-20260619184156-a40           Job completed
6m39s       Normal    Pulling                    pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    Pulling image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel"
5m39s       Normal    FileSystemResizeRequired   persistentvolumeclaim/atharv-goedelv2-storage         Require file system resize of volume on node
5m39s       Normal    Resizing                   persistentvolumeclaim/atharv-goedelv2-storage         External resizer is resizing volume pvc-9acf8bae-98a9-4ee7-a285-cfd0be779a5c
5m33s       Warning   Failed                     pod/perfseer-e2e-verify4c-20260619183732-v100-t2x9r   Failed to pull image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel": rpc error: code = Canceled desc = failed to pull and unpack image "docker.io/pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel": failed to extract layer (application/vnd.docker.image.rootfs.diff.tar.gzip sha256:3b6f6be7fc66225b79a9031da206ea0b2353262888c8b5249641d622855758b8) to overlayfs as "extract-874009933-PRNA sha256:5405cde092459f52ee2c96a87e2a3216e3f19eec8b0bba8aa53eb3c40c7f53d4": context canceled
5m33s       Warning   Failed                     pod/perfseer-e2e-verify4c-20260619183732-v100-t2x9r   Error: ErrImagePull
87s         Normal    Pulled                     pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    Successfully pulled image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" in 5m11.795s (5m11.795s including waiting). Image size: 9371344010 bytes.
87s         Normal    Created                    pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    Created container: label-sampler
86s         Normal    Started                    pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    Started container label-sampler
86s         Normal    Killing                    pod/perfseer-e2e-verify2-20260619184156-v100-hd9cs    Stopping container label-sampler
3s          Normal    Scheduled                  pod/perfseer-e2e-verify2c-20260619184849-a40-l8bmt    Successfully assigned ecepxie/perfseer-e2e-verify2c-20260619184849-a40-l8bmt to bak-hpc1.csub.edu
3s          Normal    SuccessfulCreate           job/perfseer-e2e-verify2c-20260619184849-a40          Created pod: perfseer-e2e-verify2c-20260619184849-a40-l8bmt
2s          Normal    SuccessfulCreate           job/perfseer-e2e-verify2c-20260619184849-a100         Created pod: perfseer-e2e-verify2c-20260619184849-a100-cv8gv
2s          Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify2c-20260619184849-a40-l8bmt    AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
2s          Normal    Scheduled                  pod/perfseer-e2e-verify2c-20260619184849-a100-cv8gv   Successfully assigned ecepxie/perfseer-e2e-verify2c-20260619184849-a100-cv8gv to gp-argo.usd.edu
1s          Normal    SuccessfulAttachVolume     pod/perfseer-e2e-verify2c-20260619184849-a100-cv8gv   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"


## Parallel Report

```json
{
  "gpu_pods": 2,
  "nodes": {
    "a100": "gp-argo.usd.edu",
    "a40": "bak-hpc1.csub.edu"
  },
  "start_spread_s": 0.0
}
```

## Verifier Logs

```text
{"bad_rows": 0, "files": 2, "ok_rows": 2, "rows": 2}
{"gpu_count": 2, "gpu_names": ["NVIDIA A100-PCIE-40GB", "NVIDIA A40"], "rows": 2}

```

## Jobs

```text
NAME                                          STATUS     COMPLETIONS   DURATION   AGE   CONTAINERS      IMAGES                                        SELECTOR
perfseer-e2e-verify2c-20260619184849-a100     Complete   1/1           28s        90s   label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=46067c50-0e9a-4fea-a77c-bb8930dc83af
perfseer-e2e-verify2c-20260619184849-a40      Complete   1/1           16s        91s   label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=f955ce11-caea-489d-b8a7-824bc05c5646
perfseer-e2e-verify2c-20260619184849-verify   Complete   1/1           20s        24s   verifier        python:3.11-slim                              batch.kubernetes.io/controller-uid=3a044aff-2a25-405b-af3b-feb887f84d1c

```

## Pods

```text
NAME                                                READY   STATUS      RESTARTS   AGE   IP               NODE                NOMINATED NODE   READINESS GATES
perfseer-e2e-verify2c-20260619184849-a100-cv8gv     0/1     Completed   0          90s   10.244.73.30     gp-argo.usd.edu     <none>           <none>
perfseer-e2e-verify2c-20260619184849-a40-l8bmt      0/1     Completed   0          91s   10.244.240.186   bak-hpc1.csub.edu   <none>           <none>
perfseer-e2e-verify2c-20260619184849-verify-6864c   0/1     Completed   0          24s   10.244.73.58     gp-argo.usd.edu     <none>           <none>

```
