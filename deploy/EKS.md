# Photon on EKS

## 1. Cluster

    # cluster.yaml
    apiVersion: eksctl.io/v1alpha5
    kind: ClusterConfig
    metadata:
      name: photon
      region: us-west-2
      version: "1.29"
    managedNodeGroups:
      - name: system
        instanceType: m6i.large
        desiredCapacity: 2
      - name: gpu
        instanceType: g6e.xlarge      # L40S; use p4d/p5 for A100/H100 scale
        desiredCapacity: 2
        volumeSize: 200

    eksctl create cluster -f cluster.yaml

eksctl selects the EKS GPU AMI automatically for GPU instance types.
Install the NVIDIA device plugin so pods can request nvidia.com/gpu:

    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml

## 2. Push the gateway image

    aws ecr create-repository --repository-name photon
    docker build -t <acct>.dkr.ecr.us-west-2.amazonaws.com/photon:0.1.0 .
    docker push <acct>.dkr.ecr.us-west-2.amazonaws.com/photon:0.1.0

## 3. Deploy

    kubectl apply -f deploy/k8s/vllm-big.yaml
    helm install photon deploy/helm/photon \
      --set image.repository=<acct>.dkr.ecr.us-west-2.amazonaws.com/photon \
      --set image.tag=0.1.0
    kubectl port-forward svc/photon 8080:8080
    # then run the SMOKE_TEST.md curls against localhost:8080

## 4. Monitoring

Install kube-prometheus-stack (Helm); the pod's prometheus.io/* annotations
are picked up by its default scrape config. Import
deploy/observability/grafana/dashboards/photon.json and the alert rules in
deploy/observability/alerts.yml.

## 5. Scaling — honest v1 posture

- Gateway: single replica (per-pod SQLite; see values.yaml note). CPU HPA is
  safe only after telemetry moves to Postgres.
- vLLM: replicas are scaled MANUALLY in v1. Queue-depth-based autoscaling
  (custom-metrics HPA) is a named follow-on once traffic justifies it —
  do not improvise it during a deployment.
