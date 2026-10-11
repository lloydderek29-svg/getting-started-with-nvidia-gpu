ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git \
      python3 python3-pip python3-venv python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /workspace
COPY requirements/ requirements/

ARG WITH_TORCH=false
RUN pip install --upgrade pip \
 && pip install -r requirements/base.txt \
 && pip install -r requirements/dev.txt \
 && if [ "$WITH_TORCH" = "true" ]; then \
      pip install -r requirements/torch.txt; \
    fi

ARG USERNAME=dev
ARG UID=1000
RUN (userdel -r ubuntu || true) \
 && useradd -m -u ${UID} ${USERNAME}

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${USERNAME}
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
