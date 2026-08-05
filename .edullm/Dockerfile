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
#
# botocore's `crt` EXTRA RIDES WITH boto3 BECAUSE THE CHECKSUM A CHECKPOINT WRITE
# ASKS FOR IS NOT IN botocore ITSELF. This repository is the one with a checkpoint
# contract, so it is the one this would have cost most. The platform commits a
# checkpoint by putting the payload with `ChecksumAlgorithm="CRC32C"` and then a
# `_SUCCESS` object carrying the digest S3 attested back; its reader compares the
# two and calls a checkpoint it cannot verify corrupt. botocore has no pure-Python
# CRC-32C -- `botocore.httpchecksum` adds the algorithm to its supported set only
# when `awscrt` imports -- and boto3 depends on botocore without extras, so nothing
# here ever brought it in.
#
# Found in OLMo-core's image at the end rather than the start. Run run_019fcf53 on
# 2026-08-05 pulled that image, trained twenty steps on a T4, logged every one of
# them, and died on the first line after the training loop with
# `MissingDependencyException: Using CRC32C requires an additional dependency`. A
# whole run's compute, and nothing kept. A workload here declares more than one
# attempt against a checkpoint it can resume from, so the same failure would have
# spent every one of those attempts reaching the same line.
#
# On the same pip invocation as boto3 deliberately: one resolution picks the
# botocore both agree on, where a separate later install can move botocore under
# boto3's feet. Unpinned, unlike torch above -- a different torch changes what the
# model computes, a different awscrt changes only how fast a fixed standard value
# is arrived at.
#
# TWO ASSERTIONS RATHER THAN ONE, for the same reason the CUDA assert exists beside
# the torch pin: installing a distribution is not the same as the library reaching
# it. botocore imports awscrt inside a try/except and sets HAS_CRT False on any
# failure, so an image can carry awscrt and refuse CRC32C anyway, silently. The
# first is the exact call that died -- put_object with ChecksumAlgorithm="CRC32C",
# answered by Stubber, so it needs no network, no credentials and no bucket, while
# botocore still resolves the checksum before the stub is consulted. The second is
# awscrt's answer to CRC-32C of b"123456789", which the standard fixes at
# 0xE3069283 and the platform's checkpoints.py asserts against the same constant.
RUN set -eux; \
    python -m pip install --no-cache-dir "torch==2.9.0"; \
    python -m pip install --no-cache-dir \
      "https://github.com/edu-llm/OLMo-core/archive/99e0009ed67679c90da970ec5ba439c9459e3757.tar.gz"; \
    python -m pip install --no-cache-dir \
      "edullm-data @ https://github.com/edu-llm/edullm-data/archive/refs/tags/v0.6.3.tar.gz"; \
    python -m pip install --no-cache-dir "PyYAML>=6.0" "numpy>=1.26" "boto3>=1.34" "botocore[crt]" "transformers>=4.40"; \
    python -m pip install --no-cache-dir .; \
    python -c "import torch; assert torch.__version__.startswith('2.9.0'), torch.__version__"; \
    python -c "import torch; assert torch.version.cuda, 'CUDA torch replaced by CPU wheel'"; \
    python -c "import boto3; from botocore import UNSIGNED; from botocore.config import Config; from botocore.stub import Stubber; client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED)); stub = Stubber(client); stub.add_response('put_object', {}); stub.activate(); client.put_object(Bucket='b', Key='k', Body=b'123456789', ChecksumAlgorithm='CRC32C')"; \
    python -c "from awscrt.checksums import crc32c; assert crc32c(b'123456789') == 0xE3069283, 'awscrt computes something other than CRC-32C'"; \
    python -c "import boto3, olmo_core; from olmo_core.nn.transformer import TransformerConfig; from edullm_data.read import dataset_paths; import alt_cl; print(alt_cl.__name__, TransformerConfig.__name__, dataset_paths.__module__)"; \
    python -c "from pathlib import Path; assert Path('.edullm/train_pretrain.py').is_file()"

# No CMD / ENTRYPOINT — the platform form supplies the command.
