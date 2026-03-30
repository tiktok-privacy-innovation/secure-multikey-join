## Secure Join Operations in Multi-Identifier Databases: Performance and Practicality

## Repository Layout

This is a high level overview of how the repository is laid out. Some major folders are listed below:

* [bazel/](bazel/): Configuration for SPU's use of [Bazel](https://bazel.build/).
* [psi/](psi/): Our multi-key circuit-based private set intersection protocol, built on top of [PSI's RR22 implementation](https://github.com/secretflow/psi/tree/main/psi/algorithm/rr22)
  * [cpsi](psi/cpsi.h): The main entry for the multi-key inner-join logic
  * [permute](psi/permute.h): The proposed private permutation protocol
  * [eqt](psi/eqt.h): The proposed constant-round equality testing protocol
* [libspu/](libspu/): Core C++ implementations of SPU.
  * [core/](libspu/core/): Basic data structures used in SPU.
  * [mpc/](libspu/mpc/): Various mpc protocols. This folder defines the [standard interface](libspu/mpc/apis.h)
                         different mpc protocols need to conform.
    * [cheetah/](libspu/mpc/cheetah/): An excellent semi-honest 2PC protocol.
    * [utils/](libspu/mpc/utils/): Common utilities for different mpc protocols.

## Build

### 1. Prerequisite
We prefer a Linux build. The following build has been tested on **Ubuntu 22.04**. 

```bash
# set TARGETPLATFORM='linux/arm64' if ARM CPU is used.

# Update dependencies
apt update \
&& apt upgrade -y \
&& apt install -y gcc-11 g++-11 libasan6 \
git wget curl unzip autoconf make lld-15 \
cmake ninja-build vim-common libgl1 libglib2.0-0 \
&& apt clean \
&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
&& update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 100 

# clang is required on arm64 platform
if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then apt install -y clang-15 \
    && apt clean \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 \
; fi


# amd64 is only reqiured on amd64 platform
if [ "$TARGETPLATFORM" = "linux/amd64" ] ; then apt install -y nasm ; fi

# install conda
if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then CONDA_ARCH=aarch64 ; else CONDA_ARCH=x86_64 ; fi \
  && wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh \
  && bash Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh -b \
  && rm -f Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh \
  && /root/miniconda3/bin/conda init

# Add conda to path
export PATH="/root/miniconda3/bin:${PATH}" 

# install bazel 
if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then BAZEL_ARCH=arm64 ; else BAZEL_ARCH=amd64 ; fi \
  && wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-$BAZEL_ARCH \
  && mv bazelisk-linux-$BAZEL_ARCH /usr/bin/bazel \
  && chmod +x /usr/bin/bazel
```

### 2. Run the Main Programs 

```bash
bazel run -c opt -- psi:cpsi_test --gtest_filter=PSI/CPSITest.MultipleKeys/Server131072Client1024Payload1
```

It might take some times to fetch the dependencies.

## Note
This repo is built from the [secretflow](https://github.com/secretflow) framework. 
