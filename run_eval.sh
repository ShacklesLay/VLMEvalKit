NUM_GPUS=8
models=(finetune_from_cmov3-pretrain cmov3_baseline finetune_from_cmov3-midtune_epoch10)
datasets=(MMStar BLINK RealWorldQA OCRBench MME MMMU_DEV_VAL POPE TextVQA_VAL DocVQA_VAL ChartQA_TEST)

torchrun --nproc-per-node=${NUM_GPUS} run_with_lark.py --data ${datasets[@]} --model ${models[@]} --judge exact_matching --verbose

for model in ${models[@]}; do
    python summarize_eval.py --path ./outputs/${model}
done