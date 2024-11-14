NUM_GPUS=8
models=(llava_qwen2_0.5b_baseline)
datasets=(MMStar BLINK RealWorldQA OCRBench MME MMMU_DEV_VAL POPE)

torchrun --nproc-per-node=${NUM_GPUS} run_with_lark.py --data ${datasets[@]} --model ${models[@]} --judge exact_matching --verbose