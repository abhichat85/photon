# Air-gapped deployment profile

For environments with no runtime internet egress (legal/finance mandates).

## 1. Mirror images into the customer registry

    for img in vllm/vllm-openai:latest prom/prometheus:v2.53.0 grafana/grafana:11.1.0 <photon-image>; do
      docker pull $img
      docker tag $img registry.customer.internal/$img
      docker push registry.customer.internal/$img
    done

## 2. Pre-load model weights (no HF downloads at runtime)

On a connected staging box:

    huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/qwen-14b

Transfer models/ into the environment (approved media/process), expose as a
PersistentVolume, and point vLLM at the local path with offline env:

    args: ["--model", "/models/qwen-14b", "--port", "8000"]
    env:
      - {name: HF_HUB_OFFLINE, value: "1"}
      - {name: TRANSFORMERS_OFFLINE, value: "1"}

## 3. Enforce no egress (don't just configure it — enforce it)

    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: default-deny-egress
      namespace: photon
    spec:
      podSelector: {}
      policyTypes: [Egress]
      egress:
        - to:
            - podSelector: {}          # intra-namespace only
        - to:
            - namespaceSelector:
                matchLabels:
                  kubernetes.io/metadata.name: kube-system
              podSelector:
                matchLabels:
                  k8s-app: kube-dns
          ports:
            - {protocol: UDP, port: 53}
            - {protocol: TCP, port: 53}

## 4. Verify

    kubectl exec deploy/photon -- python -c \
      "import httpx; httpx.get('https://example.com', timeout=3)"   # MUST fail
    # then run SMOKE_TEST.md curls via port-forward — MUST succeed
