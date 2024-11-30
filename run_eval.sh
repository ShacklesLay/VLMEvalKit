NUM_GPUS=8
models=(blip3kale558k blip3kale3M-v3-0.5M blip3kale3M-v3-1M)
datasets=(MMStar BLINK RealWorldQA OCRBench MME MMMU_DEV_VAL POPE)

torchrun --nproc-per-node=${NUM_GPUS} run_with_lark.py --data ${datasets[@]} --model ${models[@]} --judge exact_matching --verbose

for model in ${models[@]}; do
    python summarize_eval.py --path ./outputs/${model}
done