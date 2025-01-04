import pandas as pd
import os
import json
import argparse
import datetime
import csv

def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)
        
def main(args):
    path = args.path
    files = os.listdir(path)
    model_name = path.split('/')[-1]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    AI2D_TEST_file = [f for f in files if 'AI2D_TEST_acc' in f][0]
    df = pd.read_csv(os.path.join(path, AI2D_TEST_file))
    AI2D_TEST_file_score = df['Overall'].values[0]

    MME_file = [f for f in files if 'MME_score' in f][0]
    df = pd.read_csv(os.path.join(path, MME_file))
    MME_per = df['perception'].values[0]
    MME_re = df['reasoning'].values[0]

    DocVQA_file = [f for f in files if 'DocVQA_VAL_acc' in f][0]
    df = pd.read_csv(os.path.join(path, DocVQA_file))
    DocVQA_score = df['Overall'].values[0]

    MathVista_file = [f for f in files if 'MathVista_MINI_gpt-4o-mini_score' in f][0]
    df = pd.read_csv(os.path.join(path, MathVista_file))
    MathVista_score = df['acc'].values[0]

    OCRBench_file = [f for f in files if 'OCRBench_score' in f][0]
    data = json.load(open(os.path.join(path, OCRBench_file)))
    OCRBench_score = data['Final Score']

    MMB_file = [f for f in files if 'MMBench_dev_en_acc' in f][0]
    df = pd.read_csv(os.path.join(path, MMB_file))
    MMB_score = df['Overall'].values[0]

    RealWorldQA_file = [f for f in files if 'RealWorldQA_acc' in f][0]
    df = pd.read_csv(os.path.join(path, RealWorldQA_file))
    RealWorldQA_score = df['Overall'].values[0]
    
    NaturalBench_file = [f for f in files if 'NaturalBenchDataset_acc.csv' in f][0]
    try:
        df = pd.read_csv(os.path.join(path, NaturalBench_file))
    except:
        print(os.path.join(path, NaturalBench_file))
        raise
    NaturalBench_q_acc = df['Score'].values[0]
    NaturalBench_i_acc = df['Score'].values[1]
    NaturalBench_acc = df['Score'].values[2]
    NaturalBench_g_acc = df['Score'].values[3]

    average = (AI2D_TEST_file_score + MME_per/2000 + MME_re/800 + DocVQA_score/100 + MathVista_score/100 + OCRBench_score/1000 + MMB_score) / 7
    # 创建一个字典，将所有的分数存储在一起
    scores = {
        "timestamp": timestamp,
        "model_name": model_name,
        "MME perception": round(MME_per,3),
        "MME reasoning": round(MME_re,3),
        "MMBench": round(MMB_score,3),
        "OCRBench": round(OCRBench_score,3),
        "DocVQA": round(DocVQA_score,3),
        "MathVista": round(MathVista_score,3),
        "AI2D Test": round(AI2D_TEST_file_score,3),
        "Average": round(average,3),
        "RealWorldQA": round(RealWorldQA_score,3),
        "NaturalBench Acc": round(NaturalBench_acc,3),
        "NaturalBench Q-Acc": round(NaturalBench_q_acc,3),
        "NaturalBench I-Acc": round(NaturalBench_i_acc,3),
        "NaturalBench G-Acc": round(NaturalBench_g_acc,3),
    }


    # 将字典转换为 DataFrame
    df_scores = pd.DataFrame([scores])

    
    # 将字典转换为 DataFrame
    df_scores = pd.DataFrame([scores])

    # 保存为 CSV 文件
    output_path = '/'.join(path.split('/')[:-1]) + '/scores.csv'
    add_data_to_csv(output_path, scores)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/remote-home1/cktan/reps/VLMEvalKit/outputs/baseline_sdpa")
    args = parser.parse_args()
    
    main(args)