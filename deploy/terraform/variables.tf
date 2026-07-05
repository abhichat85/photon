variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "photon"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "gpu_instance_type" {
  description = "GPU node instance type (g6e.xlarge = L40S; p4d/p5 for A100/H100)"
  type        = string
  default     = "g6e.xlarge"
}

variable "gpu_desired_size" {
  description = "GPU node group desired size (manual scaling in v1 — see EKS.md §5)"
  type        = number
  default     = 2
}
