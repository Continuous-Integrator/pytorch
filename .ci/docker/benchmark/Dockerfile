# Extends a base CI image with TorchBench/HuggingFace/timm inductor benchmark deps.
#
# This Dockerfile is used in a two-phase docker build:
#   Phase 1: Build the base image (e.g., pytorch-linux-jammy-py3-gcc11)
#   Phase 2: Build this image FROM the base, adding only benchmark deps
#
# Usage:
#   docker build -f benchmark/Dockerfile \
#     --build-arg BASE_IMAGE=<ecr-path>:<base-image-tag> \
#     -t <benchmark-image-tag> .

ARG BASE_IMAGE

FROM ${BASE_IMAGE}

USER root

COPY ./common/install_inductor_benchmark_deps.sh install_inductor_benchmark_deps.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/huggingface-requirements.txt huggingface-requirements.txt
COPY ci_commit_pins/timm.txt timm.txt
COPY ci_commit_pins/torchbench.txt torchbench.txt

ENV BUILD_AOT_INDUCTOR_TEST=yes
RUN bash ./install_inductor_benchmark_deps.sh
RUN rm install_inductor_benchmark_deps.sh common_utils.sh timm.txt huggingface-requirements.txt torchbench.txt

USER jenkins
CMD ["bash"]
