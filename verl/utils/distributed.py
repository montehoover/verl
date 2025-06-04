# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for distributed training."""

import logging
import os

import torch.distributed

from verl.utils.device import get_torch_device, is_cuda_available

logger = logging.getLogger(__name__)


def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    torch.distributed.init_process_group("nccl" if is_cuda_available else "hccl", timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def deconflict_cuda_rocm() -> None:
    """
    If both HIP/CUDA_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES are defined
    in the current environment, unset ROCR_VISIBLE_DEVICES to prevent
    conflict in WorkerDict actors.
    """
    # Either HIP or CUDA may be used depending on how PyTorch was built.
    hip_mask   = os.environ.get("HIP_VISIBLE_DEVICES")   # ROCm-aware PyTorch
    cuda_mask  = os.environ.get("CUDA_VISIBLE_DEVICES")  # Vanilla CUDA build
    rocr_mask  = os.environ.get("ROCR_VISIBLE_DEVICES")  # Lower-level ROCm

    # Decide whether we have a high-level mask from either HIP or CUDA
    high_level_mask = hip_mask or cuda_mask

    if high_level_mask and rocr_mask:
        # Remove the low-level one; keep the high-level mask we rely on
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        logger.warning(
            "[run_ppo] Detected both %s_VISIBLE_DEVICES (%s) and "
            "ROCR_VISIBLE_DEVICES (%s). Unsetting ROCR_VISIBLE_DEVICES to "
            "prevent GPU-mask conflicts.",
            "HIP" if hip_mask else "CUDA",
            high_level_mask,
            rocr_mask,
        )