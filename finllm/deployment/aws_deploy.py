#!/usr/bin/env python3
"""
Script to deploy FinLLM to AWS
"""
import argparse
import os
import subprocess
import json
import boto3
import yaml
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run shell command and return output"""
    logger.info(f"Running command: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def docker_login_ecr(region):
    """Login to ECR"""
    account_id = run_command(f"aws sts get-caller-identity --query Account --output text")
    ecr_registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    
    # Get ECR login token
    token = run_command(f"aws ecr get-login-password --region {region}")
    
    # Login to ECR
    run_command(f"echo {token} | docker login --username AWS --password-stdin {ecr_registry}")
    
    return ecr_registry

def build_and_push_docker_image(region, image_name, tag="latest"):
    """Build and push Docker image to ECR"""
    # Get ECR registry URL
    ecr_registry = docker_login_ecr(region)
    
    # Create ECR repository if it doesn't exist
    try:
        run_command(f"aws ecr describe-repositories --repository-names {image_name} --region {region}")
    except subprocess.CalledProcessError:
        logger.info(f"Creating ECR repository: {image_name}")
        run_command(f"aws ecr create-repository --repository-name {image_name} --region {region}")
    
    # Build Docker image
    image_uri = f"{ecr_registry}/{image_name}:{tag}"
    logger.info(f"Building Docker image: {image_uri}")
    run_command(f"docker build -t {image_uri} .")
    
    # Push Docker image
    logger.info(f"Pushing Docker image: {image_uri}")
    run_command(f"docker push {image_uri}")
    
    return image_uri

def update_k8s_manifests(manifest_file, replacements):
    """Update Kubernetes manifest files with replacements"""
    logger.info(f"Updating manifest: {manifest_file}")
    
    with open(manifest_file, 'r') as f:
        content = f.read()
    
    # Perform replacements
    for key, value in replacements.items():
        content = content.replace(f"${{{key}}}", value)
    
    # Write updated content back to file
    with open(manifest_file, 'w') as f:
        f.write(content)

def deploy_to_eks(cluster_name, region, manifest_files):
    """Deploy to EKS cluster"""
    # Update kubeconfig
    logger.info(f"Updating kubeconfig for cluster: {cluster_name}")
    run_command(f"aws eks update-kubeconfig --name {cluster_name} --region {region}")
    
    # Apply manifests
    for manifest in manifest_files:
        logger.info(f"Applying manifest: {manifest}")
        run_command(f"kubectl apply -f {manifest}")

def setup_monitoring(cluster_name, region):
    """Set up monitoring with Prometheus and Grafana"""
    # Add Helm repositories
    run_command("helm repo add prometheus-community https://prometheus-community.github.io/helm-charts")
    run_command("helm repo add grafana https://grafana.github.io/helm-charts")
    run_command("helm repo update")
    
    # Install Prometheus
    logger.info("Installing Prometheus")
    run_command("helm install prometheus prometheus-community/prometheus "
                "--namespace monitoring --create-namespace "
                "--set server.persistentVolume.size=10Gi "
                "--set alertmanager.persistentVolume.size=2Gi")
    
    # Install Grafana
    logger.info("Installing Grafana")
    run_command("helm install grafana grafana/grafana "
                "--namespace monitoring "
                "--set persistence.enabled=true "
                "--set persistence.size=5Gi "
                "--set adminPassword=admin")
    
    # Get Grafana service URL
    grafana_url = run_command("kubectl get svc --namespace monitoring grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'")
    logger.info(f"Grafana URL: http://{grafana_url}")
    
    return grafana_url

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Deploy FinLLM to AWS')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--cluster', required=True, help='EKS cluster name')
    parser.add_argument('--image', default='finllm', help='Docker image name')
    parser.add_argument('--tag', default='latest', help='Docker image tag')
    parser.add_argument('--acm-cert-arn', help='ACM Certificate ARN for HTTPS')
    args = parser.parse_args()
    
    # Build and push Docker image
    image_uri = build_and_push_docker_image(args.region, args.image, args.tag)
    logger.info(f"Docker image pushed: {image_uri}")
    
    # Prepare replacements for Kubernetes manifests
    replacements = {
        'ECR_REPOSITORY_URI': image_uri.rsplit(':', 1)[0],  # Remove tag
    }
    
    if args.acm_cert_arn:
        replacements['ACM_CERTIFICATE_ARN'] = args.acm_cert_arn
    
    # Update Kubernetes manifests
    manifest_files = [
        'kubernetes/deployment.yaml',
        'kubernetes/ingress.yaml'
    ]
    
    for manifest in manifest_files:
        update_k8s_manifests(manifest, replacements)
    
    # Deploy to EKS
    deploy_to_eks(args.cluster, args.region, manifest_files)
    
    # Setup monitoring
    grafana_url = setup_monitoring(args.cluster, args.region)
    
    # Final message
    logger.info(f"Deployment completed successfully!")
    logger.info(f"Grafana URL: http://{grafana_url} (admin/admin)")
    
    # Wait for services to be ready
    logger.info("Waiting for services to be ready...")
    time.sleep(30)
    
    # Get service URL
    try:
        service_url = run_command("kubectl get ingress finllm-api-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'")
        logger.info(f"API URL: https://{service_url}")
    except:
        logger.info("Ingress not yet available. Check AWS console for the ALB URL.")

if __name__ == "__main__":
    main()