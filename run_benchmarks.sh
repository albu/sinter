#!/bin/bash
set -e

# Configuration
IMAGE_NAME="sinter-bench"
DOCKERFILE="Dockerfile.benchmark"

echo "🚀 Building benchmark container..."
docker build -t $IMAGE_NAME -f $DOCKERFILE .

echo "🔥 Running individual benchmarks..."
docker run --rm $IMAGE_NAME python/benchmarks/benchmark_unfair_individual.py

echo "📊 Running fair comparison benchmarks..."
docker run --rm $IMAGE_NAME python/benchmarks/benchmark_fair_v1_v2.py
