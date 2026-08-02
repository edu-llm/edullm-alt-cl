# Research image for the eduLLM platform build pipeline.
#
# The platform supplies the registered base image as --build-arg BASE_IMAGE, already
# pinned to a digest. Its verify_dockerfile_base gate refuses this file unless
# BASE_IMAGE is declared in the global scope with no default and every stage roots
# at ${BASE_IMAGE}.
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/edullm-alt-cl

# C compiler for torch.compile / Inductor (same rationale as OLMo-core's image).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gcc g++; \
    rm -rf /var/lib/apt/lists/*; \
    cc --version

COPY . .

# Pins match configs/*.yaml olmo_core.revision and requirements.txt edullm-data.
# Tarball URLs (no git in the bare python base). Torch first so pip does not
# re-resolve it; CUDA assert fails the build if a CPU wheel sneaks in.
RUN set -eux; \
    python -m pip install --no-cache-dir "torch==2.9.0"; \
    python -m pip install --no-cache-dir \
      "https://github.com/edu-llm/OLMo-core/archive/99e0009ed67679c90da970ec5ba439c9459e3757.tar.gz"; \
    python -m pip install --no-cache-dir \
      "edullm-data @ https://github.com/edu-llm/edullm-data/archive/refs/tags/v0.6.3.tar.gz"; \
    python -m pip install --no-cache-dir "PyYAML>=6.0" "numpy>=1.26" "boto3>=1.34" "transformers>=4.40"; \
    python -m pip install --no-cache-dir .; \
    python -c "import torch; assert torch.__version__.startswith('2.9.0'), torch.__version__"; \
    python -c "import torch; assert torch.version.cuda, 'CUDA torch replaced by CPU wheel'"; \
    python -c "import boto3, olmo_core; from olmo_core.nn.transformer import TransformerConfig; from edullm_data.read import dataset_paths; import alt_cl; print(alt_cl.__name__, TransformerConfig.__name__, dataset_paths.__module__)"; \
    python -c "from pathlib import Path; assert Path('.edullm/train_pretrain.py').is_file()"

# No CMD / ENTRYPOINT — the platform form supplies the command.
