################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import argparse
import os

import torch
import torch.distributed
import nvshmem.core

from triton_dist.kernels.nvidia import AG_MOE_RS, create_ag_group_gemm_context, ag_group_gemm
from triton_dist.kernels.nvidia.comm_perf_model import (estimate_all_gather_time_ms, get_nic_gbps_per_gpu)
from triton_dist.kernels.nvidia.gemm_perf_model import (get_dram_gbps, get_tensorcore_tflops)
from triton_dist.utils import (TP_GROUP, assert_allclose, dist_print, get_device_max_shared_memory_size,
                               get_intranode_max_speed, group_profile, initialize_distributed, perf_func, sleep_async)
from functools import partial


def estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype: torch.dtype, b_dtype: torch.dtype, stages: int):
    return (BM * BK * a_dtype.itemsize + BN * BK * b_dtype.itemsize) * (stages - 1)


def estimate_gemm_max_stages(BM, BN, BK, a_dtype, b_dtype, shared_memory_limit: int):
    return shared_memory_limit // estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype, b_dtype, 2) + 1

def perf_test(name, input_len, dtype: torch.dtype, config, debug=False):
    tp_group = TP_GROUP()
    WORLD_SIZE = tp_group.size()
    RANK = tp_group.rank()
    token_nums = input_len 
    hidden_size = config["N"]
    inter_size = config["K"]
    num_experts = config["E"]
    topk = config["TOPK"]
    local_num_tokens = token_nums // WORLD_SIZE
    local_inter_size = inter_size // WORLD_SIZE * 2

    ## generate needed tensors
    hidden_states = ((-2 * torch.rand((local_num_tokens, hidden_size), device="cuda", dtype=dtype) + 1) / 100 * (RANK + 1))
    up_weight = torch.rand([num_experts, local_inter_size, hidden_size], dtype=dtype, device="cuda")
    down_weight = torch.rand([num_experts, hidden_size, local_inter_size // 2], dtype=dtype, device="cuda")
    score = ((-2 * torch.randn((local_num_tokens, num_experts), device="cuda", dtype=dtype) + 1) / 100 * (RANK + 1))
    score = torch.softmax(score, dim=-1)

    ag_moe_rs = AG_MOE_RS(hidden_size, inter_size, num_experts, topk)
    ag_moe_rs.set_up_down_weight(up_weight, down_weight)

    name = name.lower().replace(" ", "_").replace("-", "_")
    with group_profile(f"ag_moe_{name}_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile, group=tp_group):
        sleep_async(100)
        output, duration_triton_ms = perf_func(partial(ag_moe_rs.forward, hidden_states, score), 
                                    iters=args.iters, warmup_iters=args.warmup_iters)
    
    dist_print(
        f"RANK {RANK} perf: {duration_triton_ms} ms",
        need_sync=True,
        allowed_ranks=list(range(WORLD_SIZE)),
    )
    # TODO(houqi.1993) do not release nvshmem tensor: due to a BUG from nvshmem4py
    # ctx.finalize()


layer_configs = {
    # "DeepSeek-MoE": {
    #     "N": 2048, "K": 1408, "E": 64, "TOPK": 6, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
    #     "num_warps": 8
    # },
    "Qwen3-MoE": {
        "N": 2048, "K": 6144, "E": 128, "TOPK": 8, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--autotune", default=False, action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    initialize_distributed()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    for name, config in layer_configs.items():
        perf_test(name, args.M, dtype, config, args.debug)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
