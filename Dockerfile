# Reference image for running RankFM on Linux. Build with an explicit instruction-set baseline for x86 hosts:
#   docker build --build-arg RANKFM_MARCH=x86-64-v3 -t rankfm .
FROM python:3.12-slim

ARG RANKFM_MARCH=""

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN RANKFM_MARCH="${RANKFM_MARCH}" pip install --no-cache-dir ".[blas]" scipy

# run from outside the source tree so the installed package is the one imported
WORKDIR /work
RUN cp /src/ci/container_check.py .
CMD ["python", "container_check.py"]
