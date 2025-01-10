################################################################################
ARG BASE_DOCKER=ubuntu:22.04
FROM $BASE_DOCKER AS rocm_base
################################################################################

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y python3 python-is-python3

# Install bzip2 and sqlite3 packages
RUN apt-get update && apt-get install -y \
    sqlite3 libsqlite3-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Add target file to help determine which device(s) to build for
ARG GPU_DEVICE_TARGETS="gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100"
ENV GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS}

# Install ROCm
ARG ROCM_VERSION=6.0.0
ARG ROCM_PATH=/opt/rocm-${ROCM_VERSION}
ENV ROCM_PATH=${ROCM_PATH}
ARG ROCM_BUILD_JOB
ARG ROCM_BUILD_NUM
RUN --mount=type=bind,source=build/rocm/tools/get_rocm.py,target=get_rocm.py \
    --mount=type=cache,target=/var/cache/apt \
    python3 get_rocm.py --rocm-version=$ROCM_VERSION --job-name=$ROCM_BUILD_JOB --build-num=$ROCM_BUILD_NUM

# add ROCm bins to PATH
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="/root/bin:/root/.local/bin:$PATH"

# install pyenv and python build dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
        git \
        libssl-dev \
        libffi-dev \
        libreadline-dev \
        liblzma-dev

# Install pyenv with different python versions
ARG PYTHON_VERSION=3.10.14
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install $PYTHON_VERSION
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    eval "$(pyenv init -)" && \
    pyenv local ${PYTHON_VERSION} && \
    pip3 install --upgrade --force-reinstall setuptools pip && \
    pip3 install \
        "numpy<2" \
        build \
        wheel \
        six \
        auditwheel \
        scipy \
        pytest \
        pytest-html \
        pytest_html_merger \
        pytest-reportlog \
        pytest-rerunfailures \
        pytest-json-report \
        pytest-csv \
        cloudpickle \
        portpicker \
        matplotlib \
        absl-py \
        flatbuffers \
        hypothesis

################################################################################
FROM rocm_base AS rt_build
################################################################################

ARG JAX_VERSION
ARG JAX_COMMIT
ARG XLA_COMMIT
ARG JAX_USE_CLANG

LABEL com.amdgpu.rocm_version="$ROCM_VERSION" \
      com.amdgpu.python_version="$PYTHON_VERSION" \
      com.amdgpu.jax_version="$JAX_VERSION" \
      com.amdgpu.jax_commit="$JAX_COMMIT" \
      com.amdgpu.xla_commit="$XLA_COMMIT"


# Create a directory to copy and retain the wheels in the image.
RUN mkdir -p /rocm_jax_wheels

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    --mount=type=bind,source=wheelhouse,target=/wheelhouse \
    cp /wheelhouse/* /rocm_jax_wheels/ && \
    ls -lah /wheelhouse && \
    pip3 install wheelhouse/*none*.whl wheelhouse/*jaxlib*.whl && \
    pip3 install wheelhouse/*rocm60*.whl

