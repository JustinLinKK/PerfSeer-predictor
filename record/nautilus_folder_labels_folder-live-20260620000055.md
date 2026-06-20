# Nautilus Folder Label Sampling folder-live-20260620000055

- Models dir: `/private/tmp/perfseer_folder-live-20260620000055_models`
- Local labels dir: `/Users/downeyflyfan/Research_Projects/AI/Agents/Agents_Scheduler/PerfSeer-predictor/labels/folder-live-20260620000055`
- Namespace: `ecepxie`
- Persistent Volume Claim: `test-pvc`
- GPUs: `a100, a40, l4, rtx_a4000`
- Manifest: `record/nautilus_folder_labels_folder-live-20260620000055.manifest.jsonl`
- YAML: `record/nautilus_folder_labels_folder-live-20260620000055.yaml`
- Model count: `2`

$ kubectl get jobs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
No resources found in ecepxie namespace.


$ kubectl get pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                              READY   STATUS              RESTARTS   AGE   IP       NODE                              NOMINATED NODE   READINESS GATES
perfseer-label-stage-folder-live-20260620000055   0/1     ContainerCreating   0          1s    <none>   k8s-haosu-24.sdsc.optiputer.net   <none>           <none>


$ kubectl describe pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055
Name:             perfseer-label-stage-folder-live-20260620000055
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             k8s-haosu-24.sdsc.optiputer.net/67.58.63.23
Start Time:       Sat, 20 Jun 2026 00:01:06 -0400
Labels:           app=perfseer-label-folder
                  perfseer-run-id=folder-live-20260620000055
Annotations:      kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container stage; ephemeral-storage limit for container stage
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Pending
IP:               
IPs:              <none>
Containers:
  stage:
    Container ID:  
    Image:         python:3.11-slim
    Image ID:      
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
      sleep 86400
    State:          Waiting
      Reason:       ContainerCreating
    Ready:          False
    Restart Count:  0
    Limits:
      cpu:                500m
      ephemeral-storage:  50Gi
      memory:             1Gi
    Requests:
      cpu:                100m
      ephemeral-storage:  0
      memory:             256Mi
    Environment:
      NVIDIA_VISIBLE_DEVICES:  void
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-hgcjz (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   False 
  Initialized                 True 
  Ready                       False 
  ContainersReady             False 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-hgcjz:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists
Events:
  Type    Reason                  Age   From                     Message
  ----    ------                  ----  ----                     -------
  Normal  Scheduled               2s    default-scheduler        Successfully assigned ecepxie/perfseer-label-stage-folder-live-20260620000055 to k8s-haosu-24.sdsc.optiputer.net
  Normal  SuccessfulAttachVolume  2s    attachdetach-controller  AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"


$ kubectl logs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 --all-containers=true --tail=100
Error from server (BadRequest): container "stage" in pod "perfseer-label-stage-folder-live-20260620000055" is waiting to start: ContainerCreating


$ kubectl get events -n ecepxie --sort-by=.lastTimestamp
LAST SEEN   TYPE     REASON                     OBJECT                                                MESSAGE
7m53s       Normal   Resizing                   persistentvolumeclaim/atharv-goedelv2-storage         External resizer is resizing volume pvc-9acf8bae-98a9-4ee7-a285-cfd0be779a5c
7m53s       Normal   FileSystemResizeRequired   persistentvolumeclaim/atharv-goedelv2-storage         Require file system resize of volume on node
3s          Normal   Scheduled                  pod/perfseer-label-stage-folder-live-20260620000055   Successfully assigned ecepxie/perfseer-label-stage-folder-live-20260620000055 to k8s-haosu-24.sdsc.optiputer.net
3s          Normal   SuccessfulAttachVolume     pod/perfseer-label-stage-folder-live-20260620000055   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"

## Submitted Job

- GPU: `a100`
- Job: `perfseer-label-folder-live-20260620000055-a100`
- Switched from: `none`


$ kubectl get jobs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                             STATUS    COMPLETIONS   DURATION   AGE   CONTAINERS      IMAGES                                        SELECTOR
perfseer-label-folder-live-20260620000055-a100   Running   0/1           0s         0s    label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7


$ kubectl get pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                                   READY   STATUS              RESTARTS   AGE   IP               NODE                              NOMINATED NODE   READINESS GATES
perfseer-label-folder-live-20260620000055-a100-zgzvh   0/1     ContainerCreating   0          1s    <none>           gp-argo.usd.edu                   <none>           <none>
perfseer-label-stage-folder-live-20260620000055        1/1     Running             0          18s   10.244.149.116   k8s-haosu-24.sdsc.optiputer.net   <none>           <none>


$ kubectl describe pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055
Name:             perfseer-label-folder-live-20260620000055-a100-zgzvh
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             gp-argo.usd.edu/206.209.0.17
Start Time:       Sat, 20 Jun 2026 00:01:23 -0400
Labels:           app=perfseer-label-folder
                  batch.kubernetes.io/controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
                  batch.kubernetes.io/job-name=perfseer-label-folder-live-20260620000055-a100
                  controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
                  job-name=perfseer-label-folder-live-20260620000055-a100
                  perfseer-gpu-key=a100
                  perfseer-run-id=folder-live-20260620000055
Annotations:      kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container label-sampler; ephemeral-storage limit for container label-sampler
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Pending
IP:               
IPs:              <none>
Controlled By:    Job/perfseer-label-folder-live-20260620000055-a100
Containers:
  label-sampler:
    Container ID:  
    Image:         pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
    Image ID:      
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
    Args:
      set -euo pipefail && mkdir -p /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100 && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/run_profile.py --manifest /workspace/perfseer-folder-runs/folder-live-20260620000055/manifest/manifest.jsonl --models-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/models --output-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100 --device cuda --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --sample-interval 0.01 --optimizer sgd --sm-occupancy-source nvml_proxy --precision-config fp32_ieee && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/verify_sampled_labels.py /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100
      
    State:          Waiting
      Reason:       ContainerCreating
    Ready:          False
    Restart Count:  0
    Limits:
      cpu:                2
      ephemeral-storage:  50Gi
      memory:             4Gi
      nvidia.com/a100:    1
    Requests:
      cpu:                1
      ephemeral-storage:  0
      memory:             2Gi
      nvidia.com/a100:    1
    Environment:          <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-kkshm (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   False 
  Initialized                 True 
  Ready                       False 
  ContainersReady             False 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-kkshm:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


Name:             perfseer-label-stage-folder-live-20260620000055
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             k8s-haosu-24.sdsc.optiputer.net/67.58.63.23
Start Time:       Sat, 20 Jun 2026 00:01:06 -0400
Labels:           app=perfseer-label-folder
                  perfseer-run-id=folder-live-20260620000055
Annotations:      cni.projectcalico.org/containerID: 87ec2445ec39296b16c3ba5c5013dbaca1260120cf59e79623246e3d48db7085
                  cni.projectcalico.org/podIP: 10.244.149.116/32
                  cni.projectcalico.org/podIPs: 10.244.149.116/32,fdf0:17b3:c3ec:1f79:10:0:1:f928/128
                  kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container stage; ephemeral-storage limit for container stage
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Running
IP:               10.244.149.116
IPs:
  IP:  10.244.149.116
  IP:  fdf0:17b3:c3ec:1f79:10:0:1:f928
Containers:
  stage:
    Container ID:  containerd://4e75104418d803debed102e24df272201930040b010996ca7eda355b5fed8ec4
    Image:         python:3.11-slim
    Image ID:      docker.io/library/python@sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
      sleep 86400
    State:          Running
      Started:      Sat, 20 Jun 2026 00:01:13 -0400
    Ready:          True
    Restart Count:  0
    Limits:
      cpu:                500m
      ephemeral-storage:  50Gi
      memory:             1Gi
    Requests:
      cpu:                100m
      ephemeral-storage:  0
      memory:             256Mi
    Environment:
      NVIDIA_VISIBLE_DEVICES:  void
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-hgcjz (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True 
  Initialized                 True 
  Ready                       True 
  ContainersReady             True 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-hgcjz:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


$ kubectl logs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 --all-containers=true --tail=100
Error from server (BadRequest): container "label-sampler" in pod "perfseer-label-folder-live-20260620000055-a100-zgzvh" is waiting to start: ContainerCreating


$ kubectl get events -n ecepxie --sort-by=.lastTimestamp
LAST SEEN   TYPE     REASON                     OBJECT                                                     MESSAGE
8m11s       Normal   Resizing                   persistentvolumeclaim/atharv-goedelv2-storage              External resizer is resizing volume pvc-9acf8bae-98a9-4ee7-a285-cfd0be779a5c
8m11s       Normal   FileSystemResizeRequired   persistentvolumeclaim/atharv-goedelv2-storage              Require file system resize of volume on node
21s         Normal   Scheduled                  pod/perfseer-label-stage-folder-live-20260620000055        Successfully assigned ecepxie/perfseer-label-stage-folder-live-20260620000055 to k8s-haosu-24.sdsc.optiputer.net
21s         Normal   SuccessfulAttachVolume     pod/perfseer-label-stage-folder-live-20260620000055        AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
14s         Normal   Pulled                     pod/perfseer-label-stage-folder-live-20260620000055        Container image "python:3.11-slim" already present on machine
14s         Normal   Created                    pod/perfseer-label-stage-folder-live-20260620000055        Created container: stage
14s         Normal   Started                    pod/perfseer-label-stage-folder-live-20260620000055        Started container stage
4s          Normal   Scheduled                  pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   Successfully assigned ecepxie/perfseer-label-folder-live-20260620000055-a100-zgzvh to gp-argo.usd.edu
4s          Normal   SuccessfulCreate           job/perfseer-label-folder-live-20260620000055-a100         Created pod: perfseer-label-folder-live-20260620000055-a100-zgzvh
3s          Normal   SuccessfulAttachVolume     pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"

## Submitted Job

- GPU: `a40`
- Job: `perfseer-label-folder-live-20260620000055-a40`
- Switched from: `none`


$ kubectl get jobs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                             STATUS    COMPLETIONS   DURATION   AGE   CONTAINERS      IMAGES                                        SELECTOR
perfseer-label-folder-live-20260620000055-a100   Running   0/1           6s         6s    label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
perfseer-label-folder-live-20260620000055-a40    Running   0/1           1s         1s    label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=edb9bb20-3a1b-43f8-ae6d-baf92a0288d4


$ kubectl get pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                                   READY   STATUS              RESTARTS   AGE   IP               NODE                              NOMINATED NODE   READINESS GATES
perfseer-label-folder-live-20260620000055-a100-zgzvh   0/1     ContainerCreating   0          7s    <none>           gp-argo.usd.edu                   <none>           <none>
perfseer-label-folder-live-20260620000055-a40-7rhzz    0/1     Pending             0          2s    <none>           <none>                            <none>           <none>
perfseer-label-stage-folder-live-20260620000055        1/1     Running             0          24s   10.244.149.116   k8s-haosu-24.sdsc.optiputer.net   <none>           <none>


$ kubectl describe pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055
Name:             perfseer-label-folder-live-20260620000055-a100-zgzvh
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             gp-argo.usd.edu/206.209.0.17
Start Time:       Sat, 20 Jun 2026 00:01:23 -0400
Labels:           app=perfseer-label-folder
                  batch.kubernetes.io/controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
                  batch.kubernetes.io/job-name=perfseer-label-folder-live-20260620000055-a100
                  controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
                  job-name=perfseer-label-folder-live-20260620000055-a100
                  perfseer-gpu-key=a100
                  perfseer-run-id=folder-live-20260620000055
Annotations:      kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container label-sampler; ephemeral-storage limit for container label-sampler
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Pending
IP:               
IPs:              <none>
Controlled By:    Job/perfseer-label-folder-live-20260620000055-a100
Containers:
  label-sampler:
    Container ID:  
    Image:         pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
    Image ID:      
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
    Args:
      set -euo pipefail && mkdir -p /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100 && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/run_profile.py --manifest /workspace/perfseer-folder-runs/folder-live-20260620000055/manifest/manifest.jsonl --models-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/models --output-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100 --device cuda --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --sample-interval 0.01 --optimizer sgd --sm-occupancy-source nvml_proxy --precision-config fp32_ieee && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/verify_sampled_labels.py /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100
      
    State:          Waiting
      Reason:       ContainerCreating
    Ready:          False
    Restart Count:  0
    Limits:
      cpu:                2
      ephemeral-storage:  50Gi
      memory:             4Gi
      nvidia.com/a100:    1
    Requests:
      cpu:                1
      ephemeral-storage:  0
      memory:             2Gi
      nvidia.com/a100:    1
    Environment:          <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-kkshm (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   False 
  Initialized                 True 
  Ready                       False 
  ContainersReady             False 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-kkshm:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


Name:             perfseer-label-folder-live-20260620000055-a40-7rhzz
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             <none>
Labels:           app=perfseer-label-folder
                  batch.kubernetes.io/controller-uid=edb9bb20-3a1b-43f8-ae6d-baf92a0288d4
                  batch.kubernetes.io/job-name=perfseer-label-folder-live-20260620000055-a40
                  controller-uid=edb9bb20-3a1b-43f8-ae6d-baf92a0288d4
                  job-name=perfseer-label-folder-live-20260620000055-a40
                  perfseer-gpu-key=a40
                  perfseer-run-id=folder-live-20260620000055
Annotations:      kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container label-sampler; ephemeral-storage limit for container label-sampler
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Pending
IP:               
IPs:              <none>
Controlled By:    Job/perfseer-label-folder-live-20260620000055-a40
Containers:
  label-sampler:
    Image:      pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
    Port:       <none>
    Host Port:  <none>
    Command:
      /bin/bash
      -lc
    Args:
      set -euo pipefail && mkdir -p /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a40 && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/run_profile.py --manifest /workspace/perfseer-folder-runs/folder-live-20260620000055/manifest/manifest.jsonl --models-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/models --output-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a40 --device cuda --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --sample-interval 0.01 --optimizer sgd --sm-occupancy-source nvml_proxy --precision-config fp32_ieee && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/verify_sampled_labels.py /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a40
      
    Limits:
      cpu:                2
      ephemeral-storage:  50Gi
      memory:             4Gi
      nvidia.com/a40:     1
    Requests:
      cpu:                1
      ephemeral-storage:  0
      memory:             2Gi
      nvidia.com/a40:     1
    Environment:          <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-ns9lq (ro)
      /workspace from work (rw)
Conditions:
  Type           Status
  PodScheduled   False 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-ns9lq:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


Name:             perfseer-label-stage-folder-live-20260620000055
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             k8s-haosu-24.sdsc.optiputer.net/67.58.63.23
Start Time:       Sat, 20 Jun 2026 00:01:06 -0400
Labels:           app=perfseer-label-folder
                  perfseer-run-id=folder-live-20260620000055
Annotations:      cni.projectcalico.org/containerID: 87ec2445ec39296b16c3ba5c5013dbaca1260120cf59e79623246e3d48db7085
                  cni.projectcalico.org/podIP: 10.244.149.116/32
                  cni.projectcalico.org/podIPs: 10.244.149.116/32,fdf0:17b3:c3ec:1f79:10:0:1:f928/128
                  kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container stage; ephemeral-storage limit for container stage
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Running
IP:               10.244.149.116
IPs:
  IP:  10.244.149.116
  IP:  fdf0:17b3:c3ec:1f79:10:0:1:f928
Containers:
  stage:
    Container ID:  containerd://4e75104418d803debed102e24df272201930040b010996ca7eda355b5fed8ec4
    Image:         python:3.11-slim
    Image ID:      docker.io/library/python@sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
      sleep 86400
    State:          Running
      Started:      Sat, 20 Jun 2026 00:01:13 -0400
    Ready:          True
    Restart Count:  0
    Limits:
      cpu:                500m
      ephemeral-storage:  50Gi
      memory:             1Gi
    Requests:
      cpu:                100m
      ephemeral-storage:  0
      memory:             256Mi
    Environment:
      NVIDIA_VISIBLE_DEVICES:  void
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-hgcjz (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True 
  Initialized                 True 
  Ready                       True 
  ContainersReady             True 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-hgcjz:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


$ kubectl logs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 --all-containers=true --tail=100
Error from server (BadRequest): container "label-sampler" in pod "perfseer-label-folder-live-20260620000055-a100-zgzvh" is waiting to start: ContainerCreating


$ kubectl get events -n ecepxie --sort-by=.lastTimestamp
LAST SEEN   TYPE      REASON                     OBJECT                                                     MESSAGE
8m16s       Normal    Resizing                   persistentvolumeclaim/atharv-goedelv2-storage              External resizer is resizing volume pvc-9acf8bae-98a9-4ee7-a285-cfd0be779a5c
8m16s       Normal    FileSystemResizeRequired   persistentvolumeclaim/atharv-goedelv2-storage              Require file system resize of volume on node
26s         Normal    Scheduled                  pod/perfseer-label-stage-folder-live-20260620000055        Successfully assigned ecepxie/perfseer-label-stage-folder-live-20260620000055 to k8s-haosu-24.sdsc.optiputer.net
26s         Normal    SuccessfulAttachVolume     pod/perfseer-label-stage-folder-live-20260620000055        AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
19s         Normal    Pulled                     pod/perfseer-label-stage-folder-live-20260620000055        Container image "python:3.11-slim" already present on machine
19s         Normal    Created                    pod/perfseer-label-stage-folder-live-20260620000055        Created container: stage
19s         Normal    Started                    pod/perfseer-label-stage-folder-live-20260620000055        Started container stage
9s          Normal    Scheduled                  pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   Successfully assigned ecepxie/perfseer-label-folder-live-20260620000055-a100-zgzvh to gp-argo.usd.edu
9s          Normal    SuccessfulCreate           job/perfseer-label-folder-live-20260620000055-a100         Created pod: perfseer-label-folder-live-20260620000055-a100-zgzvh
8s          Normal    SuccessfulAttachVolume     pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
4s          Normal    SuccessfulCreate           job/perfseer-label-folder-live-20260620000055-a40          Created pod: perfseer-label-folder-live-20260620000055-a40-7rhzz
1s          Warning   FailedScheduling           pod/perfseer-label-folder-live-20260620000055-a40-7rhzz    0/524 nodes are available: 1 Insufficient cpu, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint {nau ...

## GPU Switch

- Old GPU: `a40`
- Old Job: `perfseer-label-folder-live-20260620000055-a40`
- Reason: `unschedulable`
- New GPU: `l4`


## Submitted Job

- GPU: `l4`
- Job: `perfseer-label-folder-live-20260620000055-l4`
- Switched from: `a40`


$ kubectl get jobs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                             STATUS     COMPLETIONS   DURATION   AGE     CONTAINERS      IMAGES                                        SELECTOR
perfseer-label-folder-live-20260620000055-a100   Complete   1/1           2m18s      3m18s   label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
perfseer-label-folder-live-20260620000055-l4     Running    0/1           0s         0s      label-sampler   pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel   batch.kubernetes.io/controller-uid=2b4bc790-a9dd-44bc-9f85-ed052a611ddc


$ kubectl get pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055 -o wide
NAME                                                   READY   STATUS              RESTARTS   AGE     IP               NODE                              NOMINATED NODE   READINESS GATES
perfseer-label-folder-live-20260620000055-a100-zgzvh   0/1     Completed           0          3m19s   10.244.73.55     gp-argo.usd.edu                   <none>           <none>
perfseer-label-folder-live-20260620000055-l4-v9hk4     0/1     ContainerCreating   0          1s      <none>           nautilus-it-gpu03.fullerton.edu   <none>           <none>
perfseer-label-stage-folder-live-20260620000055        1/1     Running             0          3m36s   10.244.149.116   k8s-haosu-24.sdsc.optiputer.net   <none>           <none>


$ kubectl describe pods -n ecepxie -l perfseer-run-id=folder-live-20260620000055
Name:             perfseer-label-folder-live-20260620000055-a100-zgzvh
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             gp-argo.usd.edu/206.209.0.17
Start Time:       Sat, 20 Jun 2026 00:01:23 -0400
Labels:           app=perfseer-label-folder
                  batch.kubernetes.io/controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
                  batch.kubernetes.io/job-name=perfseer-label-folder-live-20260620000055-a100
                  controller-uid=604abc95-5c3f-47d9-8db9-704a99fda2e7
                  job-name=perfseer-label-folder-live-20260620000055-a100
                  perfseer-gpu-key=a100
                  perfseer-run-id=folder-live-20260620000055
Annotations:      cni.projectcalico.org/containerID: c72e3c121c538f2ca14c66bca470e9f9dfd8f462b3324c1b0c84fb83c6d3e389
                  cni.projectcalico.org/podIP: 
                  cni.projectcalico.org/podIPs: 
                  kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container label-sampler; ephemeral-storage limit for container label-sampler
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Succeeded
IP:               10.244.73.55
IPs:
  IP:           10.244.73.55
Controlled By:  Job/perfseer-label-folder-live-20260620000055-a100
Containers:
  label-sampler:
    Container ID:  containerd://eb55dadca6100415f6b743c1d4dcf434472cd409dfd7ae4a4336fe62d0458ca1
    Image:         pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
    Image ID:      docker.io/pytorch/pytorch@sha256:e0a9d9942dcabb0ceb8fd4e7bdc9dbe3e827ba783f6e9be66132ca975ddf8dc9
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
    Args:
      set -euo pipefail && mkdir -p /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100 && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/run_profile.py --manifest /workspace/perfseer-folder-runs/folder-live-20260620000055/manifest/manifest.jsonl --models-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/models --output-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100 --device cuda --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --sample-interval 0.01 --optimizer sgd --sm-occupancy-source nvml_proxy --precision-config fp32_ieee && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/verify_sampled_labels.py /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/a100
      
    State:          Terminated
      Reason:       Completed
      Exit Code:    0
      Started:      Sat, 20 Jun 2026 00:01:35 -0400
      Finished:     Sat, 20 Jun 2026 00:01:43 -0400
    Ready:          False
    Restart Count:  0
    Limits:
      cpu:                2
      ephemeral-storage:  50Gi
      memory:             4Gi
      nvidia.com/a100:    1
    Requests:
      cpu:                1
      ephemeral-storage:  0
      memory:             2Gi
      nvidia.com/a100:    1
    Environment:          <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-kkshm (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   False 
  Initialized                 True 
  Ready                       False 
  ContainersReady             False 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-kkshm:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


Name:             perfseer-label-folder-live-20260620000055-l4-v9hk4
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             nautilus-it-gpu03.fullerton.edu/209.129.60.133
Start Time:       Sat, 20 Jun 2026 00:04:41 -0400
Labels:           app=perfseer-label-folder
                  batch.kubernetes.io/controller-uid=2b4bc790-a9dd-44bc-9f85-ed052a611ddc
                  batch.kubernetes.io/job-name=perfseer-label-folder-live-20260620000055-l4
                  controller-uid=2b4bc790-a9dd-44bc-9f85-ed052a611ddc
                  job-name=perfseer-label-folder-live-20260620000055-l4
                  perfseer-gpu-key=l4
                  perfseer-run-id=folder-live-20260620000055
Annotations:      kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container label-sampler; ephemeral-storage limit for container label-sampler
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Pending
IP:               
IPs:              <none>
Controlled By:    Job/perfseer-label-folder-live-20260620000055-l4
Containers:
  label-sampler:
    Container ID:  
    Image:         pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
    Image ID:      
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
    Args:
      set -euo pipefail && mkdir -p /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/l4 && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/run_profile.py --manifest /workspace/perfseer-folder-runs/folder-live-20260620000055/manifest/manifest.jsonl --models-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/models --output-dir /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/l4 --device cuda --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --sample-interval 0.01 --optimizer sgd --sm-occupancy-source nvml_proxy --precision-config fp32_ieee && python /workspace/perfseer-folder-runs/folder-live-20260620000055/profile/verify_sampled_labels.py /workspace/perfseer-folder-runs/folder-live-20260620000055/labels/l4
      
    State:          Waiting
      Reason:       ContainerCreating
    Ready:          False
    Restart Count:  0
    Limits:
      cpu:                2
      ephemeral-storage:  50Gi
      memory:             4Gi
      nvidia.com/gpu:     1
    Requests:
      cpu:                1
      ephemeral-storage:  0
      memory:             2Gi
      nvidia.com/gpu:     1
    Environment:          <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-fhvd2 (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   False 
  Initialized                 True 
  Ready                       False 
  ContainersReady             False 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-fhvd2:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


Name:             perfseer-label-stage-folder-live-20260620000055
Namespace:        ecepxie
Priority:         0
Service Account:  default
Node:             k8s-haosu-24.sdsc.optiputer.net/67.58.63.23
Start Time:       Sat, 20 Jun 2026 00:01:06 -0400
Labels:           app=perfseer-label-folder
                  perfseer-run-id=folder-live-20260620000055
Annotations:      cni.projectcalico.org/containerID: 87ec2445ec39296b16c3ba5c5013dbaca1260120cf59e79623246e3d48db7085
                  cni.projectcalico.org/podIP: 10.244.149.116/32
                  cni.projectcalico.org/podIPs: 10.244.149.116/32,fdf0:17b3:c3ec:1f79:10:0:1:f928/128
                  kubernetes.io/limit-ranger:
                    LimitRanger plugin set: ephemeral-storage request for container stage; ephemeral-storage limit for container stage
                  nrp.ai/username: http://cilogon.org/serverE/users/505600
Status:           Running
IP:               10.244.149.116
IPs:
  IP:  10.244.149.116
  IP:  fdf0:17b3:c3ec:1f79:10:0:1:f928
Containers:
  stage:
    Container ID:  containerd://4e75104418d803debed102e24df272201930040b010996ca7eda355b5fed8ec4
    Image:         python:3.11-slim
    Image ID:      docker.io/library/python@sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -lc
      sleep 86400
    State:          Running
      Started:      Sat, 20 Jun 2026 00:01:13 -0400
    Ready:          True
    Restart Count:  0
    Limits:
      cpu:                500m
      ephemeral-storage:  50Gi
      memory:             1Gi
    Requests:
      cpu:                100m
      ephemeral-storage:  0
      memory:             256Mi
    Environment:
      NVIDIA_VISIBLE_DEVICES:  void
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-hgcjz (ro)
      /workspace from work (rw)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True 
  Initialized                 True 
  Ready                       True 
  ContainersReady             True 
  PodScheduled                True 
Volumes:
  work:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  test-pvc
    ReadOnly:   false
  kube-api-access-hgcjz:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:PreferNoSchedule op=Exists


$ kubectl logs -n ecepxie -l perfseer-run-id=folder-live-20260620000055 --all-containers=true --tail=100
live_test_cnn::fp32_ieee: ok
live_test_mlp::fp32_ieee: ok
{"bad_rows": 0, "files": 1, "ok_rows": 2, "rows": 2}
Error from server (BadRequest): container "label-sampler" in pod "perfseer-label-folder-live-20260620000055-l4-v9hk4" is waiting to start: ContainerCreating


$ kubectl get events -n ecepxie --sort-by=.lastTimestamp
LAST SEEN   TYPE      REASON                     OBJECT                                                     MESSAGE
3m39s       Normal    Scheduled                  pod/perfseer-label-stage-folder-live-20260620000055        Successfully assigned ecepxie/perfseer-label-stage-folder-live-20260620000055 to k8s-haosu-24.sdsc.optiputer.net
3m39s       Normal    SuccessfulAttachVolume     pod/perfseer-label-stage-folder-live-20260620000055        AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
3m32s       Normal    Started                    pod/perfseer-label-stage-folder-live-20260620000055        Started container stage
3m32s       Normal    Created                    pod/perfseer-label-stage-folder-live-20260620000055        Created container: stage
3m32s       Normal    Pulled                     pod/perfseer-label-stage-folder-live-20260620000055        Container image "python:3.11-slim" already present on machine
3m22s       Normal    Scheduled                  pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   Successfully assigned ecepxie/perfseer-label-folder-live-20260620000055-a100-zgzvh to gp-argo.usd.edu
3m22s       Normal    SuccessfulCreate           job/perfseer-label-folder-live-20260620000055-a100         Created pod: perfseer-label-folder-live-20260620000055-a100-zgzvh
3m21s       Normal    SuccessfulAttachVolume     pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
3m17s       Normal    SuccessfulCreate           job/perfseer-label-folder-live-20260620000055-a40          Created pod: perfseer-label-folder-live-20260620000055-a40-7rhzz
3m10s       Normal    Started                    pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   Started container label-sampler
3m10s       Normal    Created                    pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   Created container: label-sampler
3m10s       Normal    Pulled                     pod/perfseer-label-folder-live-20260620000055-a100-zgzvh   Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
2m16s       Warning   FailedScheduling           pod/perfseer-label-folder-live-20260620000055-a40-7rhzz    0/524 nodes are available: 1 Insufficient cpu, 1 node(s) had untolerated taint {nautilus.io/issue: 1368}, 1 node(s) had untolerated taint {nautilus.io/issue: 1574}, 1 node(s) had untolerated taint {nautilus.io/issue: 1593-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1607}, 1 node(s) had untolerated taint {nautilus.io/issue: 1641}, 1 node(s) had untolerated taint {nautilus.io/issue: 1661-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1679-1685}, 1 node(s) had untolerated taint {nautilus.io/issue: 1692}, 1 node(s) had untolerated taint {nautilus.io/issue: 1693}, 1 node(s) had untolerated taint {nautilus.io/issue: 1696}, 1 node(s) had untolerated taint {nautilus.io/issue: slow-network}, 1 node(s) had untolerated taint {nautilus.io/issue: testing}, 1 node(s) had untolerated taint {nautilus.io/reservation: internet2}, 1 node(s) had untolerated taint {nautilus.io/reservation: nrp-llm}, 1 node(s) had untolerated taint {nautilus.io/reservation: sage}, 1 node(s) had untolerated taint {nau ...
89s         Normal    Resizing                   persistentvolumeclaim/atharv-goedelv2-storage              External resizer is resizing volume pvc-9acf8bae-98a9-4ee7-a285-cfd0be779a5c
89s         Normal    FileSystemResizeRequired   persistentvolumeclaim/atharv-goedelv2-storage              Require file system resize of volume on node
63s         Normal    Completed                  job/perfseer-label-folder-live-20260620000055-a100         Job completed
4s          Normal    SuccessfulCreate           job/perfseer-label-folder-live-20260620000055-l4           Created pod: perfseer-label-folder-live-20260620000055-l4-v9hk4
3s          Normal    SuccessfulAttachVolume     pod/perfseer-label-folder-live-20260620000055-l4-v9hk4     AttachVolume.Attach succeeded for volume "pvc-dd41bf1c-2a55-467a-aaeb-73f47269e9d8"
1s          Normal    Pulled                     pod/perfseer-label-folder-live-20260620000055-l4-v9hk4     Container image "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel" already present on machine
1s          Normal    Created                    pod/perfseer-label-folder-live-20260620000055-l4-v9hk4     Created container: label-sampler

## Controller States

```json
{
  "a100": "succeeded",
  "a40": "failed",
  "l4": "pending",
  "rtx_a4000": "pending"
}
```

## Verifier Logs

```text
{"bad_rows": 0, "files": 1, "ok_rows": 2, "rows": 2}

```

## Local Output Verification

- Local labels directory: `labels/folder-live-20260620000055`
- Result file: `labels/folder-live-20260620000055/a100/results_shard0.jsonl`
- Label files:
  - `labels/folder-live-20260620000055/a100/label/label/live_test_cnn_fp32_ieee.txt`
  - `labels/folder-live-20260620000055/a100/label/label/live_test_mlp_fp32_ieee.txt`
- Local verifier command: `python3 scripts/verify_sampled_labels.py labels/folder-live-20260620000055`
- Local verifier output: `{"bad_rows": 0, "files": 1, "ok_rows": 2, "rows": 2}`
- Model rows:
  - `live_test_cnn`: `status=ok`, `gpu_name=NVIDIA A100-PCIE-40GB`
  - `live_test_mlp`: `status=ok`, `gpu_name=NVIDIA A100-PCIE-40GB`
- Kubernetes cleanup check:
  - Jobs: `No resources found in ecepxie namespace.`
  - Pods: `No resources found in ecepxie namespace.`
