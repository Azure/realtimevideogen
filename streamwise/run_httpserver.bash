#!/usr/bin/env bash

echo "Storage:"
df -h

echo ""
echo "Memory:"
free -h

echo "CPU:"
lscpu

echo ""
echo "Environment variables:"
printenv

# Auto-detect SSL certificates mounted at /certs/ (K8s TLS Secret or build-time embedded)
CERT_ARGS=()
if [[ -f "/certs/tls.crt" ]] && [[ -f "/certs/tls.key" ]]; then
    echo "HTTPS enabled: /certs/tls.crt"
    CERT_ARGS=(--certfile /certs/tls.crt --keyfile /certs/tls.key --use-https)
elif [[ -f "/certs/cert.pem" ]] && [[ -f "/certs/key.pem" ]]; then
    echo "HTTPS enabled: /certs/cert.pem"
    CERT_ARGS=(--certfile /certs/cert.pem --keyfile /certs/key.pem --use-https)
fi

python3 streamwise.py \
${CERT_ARGS[@]+"${CERT_ARGS[@]}"} \
"$@"
