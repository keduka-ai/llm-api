#!/bin/bash

# API gateway mode: stateless proxy to llama-server backends
echo "Starting API gateway (workers=4, threads=2)"
gunicorn project.wsgi:application \
    -b 0.0.0.0:8000 \
    --workers 4 \
    --threads 2 \
    --timeout 3600 \
    --worker-tmp-dir /dev/shm
