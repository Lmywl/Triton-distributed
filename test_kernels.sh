#!/bin/bash

# ### 1.gemm reducescatter test  pass √
# echo "GEMM ReduceScatter correctness test"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# # M_VALUES=(64 128 256 512 1024 2048 4096 8192)
# M_VALUES=(512)
# N=29568
# K=8192
# DTYPE='float16'

# for M in "${M_VALUES[@]}"; do
#     echo "***************Test with TP_SIZE=4 M=$M N=$N K=$K dtype=$DTYPE****************"
#     bash ./scripts/launch.sh 4 ./python/triton_dist/test/nvidia/test_gemm_rs.py $M $N $K --dtype $DTYPE
# done


# ### 2.all gather gemm test  pass √
# echo "allgather gemm correctness test"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# # shapes=("Qwen2-72B1" "Qwen2-72B2" "Qwen2-72B3" "Qwen2-72B4" "Qwen2-72B5" "Qwen2-72B6" "Qwen2-72B7" "Qwen2-72B8")
# shapes=("Qwen2-72B8")
# for shape in "${shapes[@]}"; do
#     echo "Test with TP_SIZE=4 $shape" 
#     bash ./scripts/launch.sh 4 ./python/triton_dist/test/nvidia/test_ag_gemm.py --shape_id $shape --case 'correctness'
# done


# ### 3.moe reduce scatter test:  the result is not correct, but it can run successfully
# echo "MoE reduce_scatter correctness test"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # M_VALUES=(32 64 128 256 512 1024 2048 4096 8192)
# M_VALUES=(128)
# N=2048
# K=1408
# EXPERTS=64
# TOPK=8

# for M in "${M_VALUES[@]}"; do
#     echo "***************Test with TP_SIZE=8 M=$M N=$N K=$K num_experts=$EXPERTS Topk=$TOPK****************"
#     bash ./scripts/launch.sh 8 ./python/triton_dist/test/nvidia/test_moe_reduce_rs.py $M $N $K $EXPERTS $TOPK --autotune
# done

# ### 4. allgather moe test  pass √
# echo "allgather MoE correctness test"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # shapes=(128 256 512 1024 2048 4096 8192)
# shapes=(512)
# for shape in "${shapes[@]}"; do
#     echo "Test with TP_SIZE=8 $shape" 
#     bash ./scripts/launch.sh 8 ./python/triton_dist/test/nvidia/test_ag_moe.py --M $shape 
# done

### 5. ag_moe_rs test
echo "allgather MoE Reducescatter test"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# shapes=(128 256 512 1024 2048 4096 8192)
shapes=(512)
for shape in "${shapes[@]}"; do
    echo "Test with TP_SIZE=8 $shape" 
    bash ./scripts/launch.sh 8 ./python/triton_dist/test/nvidia/test_ag_moe_rs.py --M $shape
done