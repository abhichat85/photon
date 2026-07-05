# Photon on AKS

## 1. Cluster

    az group create --name photon --location westus3
    az aks create \
      --resource-group photon --name photon \
      --node-count 1 --node-vm-size Standard_D4s_v5 \
      --generate-ssh-keys

Add a GPU node pool (NC-series; NVadsA10 = the L40S-class analogue, use
ND-series for A100/H100 scale):

    az aks nodepool add \
      --resource-group photon --cluster-name photon \
      --name gpu --node-count 2 \
      --node-vm-size Standard_NV36ads_A10_v5 \
      --node-osdisk-size 200

    az aks get-credentials --resource-group photon --name photon

Install the NVIDIA device plugin so pods can request nvidia.com/gpu:

    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml

## 2. Push the gateway image

    az acr create --resource-group photon --name photonacr --sku Standard
    az acr login --name photonacr
    docker build -t photonacr.azurecr.io/photon:0.1.0 .
    docker push photonacr.azurecr.io/photon:0.1.0

## 3. Deploy

    kubectl apply -f deploy/k8s/vllm-big.yaml
    helm install photon deploy/helm/photon \
      --set image.repository=photonacr.azurecr.io/photon \
      --set image.tag=0.1.0
    kubectl port-forward svc/photon 8080:8080
    # then run the SMOKE_TEST.md curls against localhost:8080

## 4. Monitoring

Install kube-prometheus-stack (Helm); the pod's prometheus.io/* annotations
are picked up by its default scrape config. Import
deploy/observability/grafana/dashboards/photon.json and the alert rules in
deploy/observability/alerts.yml. Azure Monitor is the managed alternative;
the promtail/Loki stack in docker-compose is the portable one.

## 5. Scaling — honest v1 posture

- Gateway: single replica (per-pod SQLite; see values.yaml note). CPU HPA is
  safe only after telemetry moves to Postgres.
- vLLM: replicas are scaled MANUALLY in v1. Queue-depth-based autoscaling
  (custom-metrics HPA) is a named follow-on — do not improvise it during a
  deployment.
