# pouwchain node + PyTorch worker (CPU) in one image.
#   docker build -t pauwchain .
# Both stages use the same Debian release, so the node binary links against the same glibc it was built with.

FROM python:3.12-slim-bookworm AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt .
COPY node/ node/
# FetchContent downloads nlohmann/json, PicoSHA2, cpp-httplib and Monocypher here.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j --target node pouw-cli \
 && strip build/node build/pouw


FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Only what the worker imports; versions come from requirements.txt. torch/torchvision are the CPU builds.
COPY requirements.txt /tmp/requirements.txt
RUN grep -E '^(numpy|safetensors)==' /tmp/requirements.txt > /tmp/base.txt \
 && grep -E '^(torch|torchvision)==' /tmp/requirements.txt > /tmp/torch.txt \
 && pip install -r /tmp/base.txt \
 && pip install -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && rm /tmp/*.txt

RUN useradd --create-home --uid 10001 pouw \
 && mkdir -p /app/data /app/state \
 && chown pouw:pouw /app/data /app/state
WORKDIR /app
COPY --from=build /src/build/node /usr/local/bin/node
COPY --from=build /src/build/pouw /usr/local/bin/pouw
COPY worker/ worker/
COPY testnet/ testnet/
USER pouw

# Ports inside the container: P2P 9100, HTTP API 9200 (see compose.yaml).
EXPOSE 9100 9200
CMD ["node"]
