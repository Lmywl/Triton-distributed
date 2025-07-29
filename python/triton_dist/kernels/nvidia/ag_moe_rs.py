import argparse
import os

import torch
import torch.distributed
import nvshmem.core
from triton_dist.utils import TP_GROUP, get_device_max_shared_memory_size
from triton_dist.kernels.nvidia import (ag_group_gemm, create_ag_group_gemm_context, gated_silu)
from triton_dist.kernels.nvidia.moe_reduce_rs import moe_reduce_rs_colwise, select_experts
from triton_dist.kernels.nvidia import (create_moe_rs_context, create_moe_rs_context_colwise, moe_reduce_rs_rowise)

def estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype: torch.dtype, b_dtype: torch.dtype, stages: int):
    return (BM * BK * a_dtype.itemsize + BN * BK * b_dtype.itemsize) * (stages - 1)
    
def estimate_gemm_max_stages(BM, BN, BK, a_dtype, b_dtype, shared_memory_limit: int):
    return shared_memory_limit // estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype, b_dtype, 2) + 1


class AllGatherMoe(torch.nn.Module):
    def __init__(self,
                local_inter_size: int,
                hidden_size: int,
                num_experts: int,
                topk: int,
                dtype: torch.dtype,
                RANK: int,
                WORLD_SIZE: int,
                LOCAL_WORLD_SIZE: int,
                max_num_tokens: int = 16 * 1024,
                BLOCK_SIZE_M: int = 128,
                BLOCK_SIZE_N: int = 256,
                BLOCK_SIZE_K: int = 64,
                GROUP_SIZE_M: int = 8,
                stages: int = 4,
                num_warps: int = 8,
                ) -> None:
        super(AllGatherMoe, self).__init__()
        BM, BN, BK = BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        shared_memory_limit = get_device_max_shared_memory_size(torch.cuda.current_device())
        max_stages = estimate_gemm_max_stages(BM, BN, BK, dtype, dtype, shared_memory_limit)
        if stages > max_stages:
            stages = max_stages
        self.ctx = create_ag_group_gemm_context(
                    max_ntokens = max_num_tokens,
                    N_per_rank = local_inter_size,
                    K = hidden_size,  
                    num_experts = num_experts,
                    topk = topk,
                    dtype = dtype,
                    rank = RANK,
                    num_ranks = WORLD_SIZE,
                    num_local_ranks = LOCAL_WORLD_SIZE,
                    BLOCK_SIZE_M = BLOCK_SIZE_M,
                    BLOCK_SIZE_N = BLOCK_SIZE_N,
                    BLOCK_SIZE_K = BLOCK_SIZE_K,
                    GROUP_SIZE_M = GROUP_SIZE_M,
                    stages = stages,
                    num_warps = num_warps,
                )
    
    def forward(self, 
                local_hidden_states: torch.Tensor, # [local_num_tokens, hidden_size]
                up_weight: torch.Tensor,           # [num_experts, hidden_size, local_inter_size]
                full_topk_ids: torch.Tensor        # [num_tokens, topk]
                )-> torch.Tensor:
        
        inter_states = ag_group_gemm(local_hidden_states, up_weight, self.ctx, full_topk_ids)
        gated_silu_states = gated_silu(inter_states, self.ctx.BLOCK_N)
        return gated_silu_states


class MoEReduceRSTensorParallel(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        local_world_size: int,
        hidden_dim: int,
        intermediate_size: int,  # origin intermediate_size 
        num_experts: int,
        topk: int,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        moe_block_size=128,
    ):
        super(MoEReduceRSTensorParallel, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.local_world_size = local_world_size
        self.local_rank = self.rank % self.local_world_size
        self.max_token_num = max_token_num
        assert (max_token_num % self.world_size == 0)
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.world_size == 0)
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.moe_block_size = moe_block_size

        self.ctx_colwise = create_moe_rs_context_colwise(
            self.rank,
            self.world_size,
            self.local_world_size,
            self.max_token_num,
            self.hidden_dim,
            self.num_experts,
            self.topk,
            self.input_dtype,
        )

    def forward(self, 
                intermediate_states: torch.Tensor,  # [topk * num_tokens, local_inter_size / 2 (self.intermediate_size_per_rank)]
                down_weight: torch.Tensor,          # [num_experts, local_inter_size / 2 , hidden_size]
                full_topk_ids: torch.Tensor, 
                full_topk_weight: torch.Tensor,
                ) -> torch.Tensor:
        
        output = moe_reduce_rs_colwise(
            intermediate_states,
            down_weight,
            full_topk_ids,
            full_topk_weight,
            ctx=self.ctx_colwise,
            n_chunks=4,
        )
        return output


class AG_MOE_RS(torch.nn.Module):
    def __init__(self,    
                hidden_size: int,     
                intermediate_size: int,         
                num_experts: int,      
                topk: int,     
                ) -> None:
        super(AG_MOE_RS, self).__init__()
        self.tp_group = TP_GROUP()
        self.rank = self.tp_group.rank()
        self.world_size = self.tp_group.size()
        self.local_world_size = self.world_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.world_size == 0)
        local_inter_size = 2 * intermediate_size // self.world_size
        self.agmoe = AllGatherMoe(
            local_inter_size=local_inter_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            topk=topk,
            dtype=torch.float16,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            LOCAL_WORLD_SIZE=self.local_world_size
            )
        
        self.moe_rs = MoEReduceRSTensorParallel(
                pg=self.tp_group,
                local_world_size=self.local_world_size,
                hidden_dim=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                topk=topk,
                )
    
    def set_up_down_weight(self, 
                           up_weight: torch.Tensor,   #[num_experts, local_inter_size, hidden_size]
                           down_weight: torch.Tensor  #[num_experts, hidden_size, local_inter_size / 2]
                           ) -> None:
        self.up_weight = up_weight.transpose(1, 2)
        self.down_weight = down_weight.transpose(1, 2)

    def forward(self,
                hidden_states: torch.Tensor,     #[local_num_tokens, hidden_size]
                router_logits: torch.Tensor,     #[local_num_tokens, num_experts]
                ) -> torch.Tensor:
        full_topk_ids, full_topk_weight = select_experts(self.tp_group, self.world_size, self.topk,
                                          router_logits.dtype, torch.device("cuda"), router_logits)
        
        inter_states = self.agmoe.forward(hidden_states, self.up_weight, full_topk_ids.to(torch.int64))
        output = self.moe_rs.forward(inter_states, self.down_weight, full_topk_ids, full_topk_weight)
        return output
