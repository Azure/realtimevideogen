#!/usr/bin/env bash
# Delete the Helm deployment and clean up resources
helm uninstall gpu-services -n rtgen
kubectl get pods -n rtgen | awk '/Terminating/ {print $1}' |  xargs -I {} kubectl delete pod {} -n rtgen --grace-period=0 --force