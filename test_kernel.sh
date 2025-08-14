#!/bin/bash
### triton_dist mega kernel test
echo "triton_dist mega kernel test"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVSHMEM_DISABLE_CUDA_VMM=0
# shapes=(16 32 64 128 256 512 1024)
shapes=(32)
for shape in "${shapes[@]}"; do
    echo "Test with TP_SIZE=8 M=$shape" 
    bash ./scripts/launch.sh 8 ./python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --seq_len $shape
done
