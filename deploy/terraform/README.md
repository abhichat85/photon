# Photon Terraform (EKS)

Infrastructure-as-code for the Photon EKS deployment — the reproducible
alternative to the `eksctl` quickstart in `../EKS.md`. Provisions VPC + EKS
control plane + a system node group + a GPU node group for vLLM.

## Use

    cd deploy/terraform
    terraform init
    terraform apply \
      -var="region=us-west-2" \
      -var="gpu_instance_type=g6e.xlarge" \
      -var="gpu_desired_size=2"

    # then wire kubectl (the apply prints this command):
    aws eks update-kubeconfig --name photon --region us-west-2

    # NVIDIA device plugin + app deploy — identical to EKS.md §1 (device plugin)
    # and §3 (helm install photon).

## Scope (honest v1)

- GPU node group is **fixed size** (`min == max == desired`) — matches the
  manual-scaling posture in EKS.md §5. Wire a Cluster Autoscaler / Karpenter
  only when traffic justifies it.
- This is the AWS reference. GKE/AKS IaC would follow the same shape with the
  `terraform-google-modules/kubernetes-engine` and `Azure/aks` modules; the
  runbooks in GKE.md / AKS.md cover the CLI path for those clouds today.
- State is local by default. For team use, add an S3 + DynamoDB backend before
  the first real apply.
