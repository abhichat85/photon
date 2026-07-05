# Photon on GKE

## 1. Cluster

    gcloud container clusters create photon \
      --region us-central1 \
      --release-channel stable \
      --num-nodes 1 \
      --machine-type e2-standard-4

Add a GPU node pool (L4 = the GKE analogue of L40S-class; use a2/a3 for
A100/H100 scale):

    gcloud container node-pools create gpu \
      --cluster photon --region us-central1 \
      --machine-type g2-standard-8 \
      --accelerator type=nvidia-l4,count=1 \
      --num-nodes 2 --disk-size 200

Install the NVIDIA device plugin (GKE ships a managed DaemonSet):

    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

## 2. Push the gateway image

    gcloud artifacts repositories create photon --repository-format=docker --location=us-central1
    docker build -t us-central1-docker.pkg.dev/<project>/photon/photon:0.1.0 .
    docker push us-central1-docker.pkg.dev/<project>/photon/photon:0.1.0

## 3. Deploy

    kubectl apply -f deploy/k8s/vllm-big.yaml
    helm install photon deploy/helm/photon \
      --set image.repository=us-central1-docker.pkg.dev/<project>/photon/photon \
      --set image.tag=0.1.0
    kubectl port-forward svc/photon 8080:8080
    # then run the SMOKE_TEST.md curls against localhost:8080

## 4. Monitoring

Install kube-prometheus-stack (Helm); the pod's prometheus.io/* annotations
are picked up by its default scrape config. Import
deploy/observability/grafana/dashboards/photon.json and the alert rules in
deploy/observability/alerts.yml. For logs, GKE integrates with Cloud Logging;
the promtail/Loki stack in docker-compose is the portable alternative.

## 5. Scaling — honest v1 posture

- Gateway: single replica (per-pod SQLite; see values.yaml note). CPU HPA is
  safe only after telemetry moves to Postgres.
- vLLM: replicas are scaled MANUALLY in v1. Queue-depth-based autoscaling
  (custom-metrics HPA) is a named follow-on — do not improvise it during a
  deployment.
