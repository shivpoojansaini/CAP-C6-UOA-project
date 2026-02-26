#!/bin/bash
# Run script for Lightning AI

PORT=${PORT:-8080}
echo "Starting Streamlit on port $PORT"

streamlit run test_streamlit.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --logger.level=debug
