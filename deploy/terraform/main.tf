# Photon EKS infrastructure as code — the Terraform alternative to the eksctl
# quickstart in deploy/EKS.md. Provisions a VPC, an EKS control plane, a small
# system node group, and a GPU node group for vLLM. Apply, then install the
# NVIDIA device plugin and `helm install photon` exactly as EKS.md §3 describes.
#
#   terraform init && terraform apply
#   aws eks update-kubeconfig --name <cluster_name> --region <region>

data "aws_availability_zones" "available" {
  state = "available"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.8"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    system = {
      instance_types = ["m6i.large"]
      min_size       = 2
      max_size       = 2
      desired_size   = 2
    }
    gpu = {
      # EKS auto-selects the GPU-optimized AMI for GPU instance types.
      instance_types = [var.gpu_instance_type]
      min_size       = var.gpu_desired_size
      max_size       = var.gpu_desired_size # manual scaling in v1 (EKS.md §5)
      desired_size   = var.gpu_desired_size
      disk_size      = 200
      labels         = { workload = "gpu" }
    }
  }
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "configure_kubectl" {
  value = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
}
