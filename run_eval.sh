NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
models=(llava_qwen2_0.5b_midtune_ve-lr_2e-5)
datasets=(MMStar BLINK RealWorldQA OCRBench MME MMMU_DEV_VAL POPE)

torchrun --nproc-per-node=${NUM_GPUS} run_with_lark.py --data ${datasets[@]} --model ${models[@]} --judge exact_matching --verbose