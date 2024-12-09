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
    
    ChartQA_TEST_file = [f for f in files if 'ChartQA_TEST_acc' in f][0]
    df = pd.read_csv(os.path.join(path, ChartQA_TEST_file))
    ChartQA_TEST_score = df['Overall'].values[0]
    
    DocVQA_VAL_file = [f for f in files if 'DocVQA_VAL_acc' in f][0]
    df = pd.read_csv(os.path.join(path, DocVQA_VAL_file))
    DocVQA_VAL_score = df['Overall'].values[0]
    
    TextVQA_VAL_file = [f for f in files if 'TextVQA_VAL_acc' in f][0]
    df = pd.read_csv(os.path.join(path, TextVQA_VAL_file))
    TextVQA_VAL_score = df['Overall'].values[0]

    average = (blink_score + MME_per/2000 + MME_re/800 + MMMU_val + MMStar_score + OCRBench_score/1000 + POPE_score/100 + RealWorldQA_score + ChartQA_TEST_score/100 + DocVQA_VAL_score/100 + TextVQA_VAL_score/100) / 11
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
        "ChartQA_TEST": round(ChartQA_TEST_score,3),
        "DocVQA_VAL": round(DocVQA_VAL_score,3),
        "TextVQA_VAL": round(TextVQA_VAL_score,3),
        "Average": round(average,3)
    }


    # 将字典转换为 DataFrame
    df_scores = pd.DataFrame([scores])

    # 保存为 CSV 文件
    output_path = os.path.join(path, "0_scores.csv") 
    df_scores.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/image_data/cktan/reps/VLMEvalKit/outputs/llava_qwen2_0.5b_midtune_baseline")
    args = parser.parse_args()
    
    main(args)