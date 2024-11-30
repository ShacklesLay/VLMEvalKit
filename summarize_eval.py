import pandas as pd
import os
import json
import argparse
def main(args):
    path = args.path
    files = os.listdir(path)

    blink_file = [f for f in files if 'BLINK_acc' in f][0]
    df = pd.read_csv(os.path.join(path, blink_file))
    blink_score = df['Overall'].values[0]

    MME_file = [f for f in files if 'MME_score' in f][0]
    df = pd.read_csv(os.path.join(path, MME_file))
    MME_per = df['perception'].values[0]
    MME_re = df['reasoning'].values[0]

    MMMU_file = [f for f in files if 'MMMU_DEV_VAL_acc' in f][0]
    df = pd.read_csv(os.path.join(path, MMMU_file))
    MMMU_val = df[df['split']=='validation']['Overall'].values[0]

    MMStar_file = [f for f in files if 'MMStar_acc' in f][0]
    df = pd.read_csv(os.path.join(path, MMStar_file))
    MMStar_score = df['Overall'].values[0]

    OCRBench_file = [f for f in files if 'OCRBench_score' in f][0]
    data = json.load(open(os.path.join(path, OCRBench_file)))
    OCRBench_score = data['Final Score']

    POPE_file = [f for f in files if 'POPE_score' in f][0]
    df = pd.read_csv(os.path.join(path, POPE_file))
    POPE_score = df[df['split']=='Overall']['Overall'].values[0]

    RealWorldQA_file = [f for f in files if 'RealWorldQA_acc' in f][0]
    df = pd.read_csv(os.path.join(path, RealWorldQA_file))
    RealWorldQA_score = df['Overall'].values[0]

    average = (blink_score + MME_per/2000 + MME_re/800 + MMMU_val + MMStar_score + OCRBench_score/1000 + POPE_score/100 + RealWorldQA_score) / 8
    # 创建一个字典，将所有的分数存储在一起
    scores = {
        "MMStar": round(MMStar_score,3),
        "BLINK": round(blink_score,3),
        "RealWorldQA": round(RealWorldQA_score,3),
        "OCRBench": round(OCRBench_score,3),
        "MME perception": round(MME_per,3),
        "MME reasoning": round(MME_re,3),
        "MMMU validation": round(MMMU_val,3),
        "POPE": round(POPE_score,3),
        "Average": round(average,3)
    }


    # 将字典转换为 DataFrame
    df_scores = pd.DataFrame([scores])

    # 保存为 CSV 文件
    output_path = os.path.join(path, "0_scores.csv") 
    df_scores.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/remote-home1/cktan/reps/VLMEvalKit/outputs/baseline_sdpa")
    args = parser.parse_args()
    
    main(args)