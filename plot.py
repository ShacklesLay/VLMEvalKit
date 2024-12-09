import matplotlib.pyplot as plt
import sys
sys.path.append('/home/image_data/cktan/reps/server_tools')
from scripts import load
import os
import numpy as np

def compute_average(data):
    blink_score = data['BLINK']
    OCRBench_score = data['OCRBench']
    POPE_score = data['POPE']
    RealWorldQA_score = data['RealWorldQA']
    MME_per = data['MME perception']
    MME_re = data['MME reasoning']
    MMMU_val = data['MMMU validation']
    MMStar_score = data['MMStar']
    return (blink_score + MME_per/2000 + MME_re/800 + MMMU_val + MMStar_score + OCRBench_score/1000 + POPE_score/100 + RealWorldQA_score) / 8 

output_dir = "./outputs"
paths = [i for i in os.listdir(output_dir) if i.startswith('blip558k_epoch')]
paths = sorted(paths, key=lambda x: float(x[-1]))
paths = [os.path.join(output_dir, i) for i in paths]

data_plot = {}
for p in paths:
    data = load(os.path.join(p, '0_scores.csv'))
    for k, v in data.items():
        if k not in data_plot:
            data_plot[k] = []
        data_plot[k].append(v)
        
baseline_data = {
        "MMStar": 0.353,
        "BLINK": 0.359,
        "RealWorldQA": 0.452,
        "OCRBench": 211,
        "MME perception": 1044,
        "MME reasoning": 251,
        "MMMU validation": 0.31,
        "POPE": 84.43,
    }
baseline_data['Average'] = compute_average(baseline_data)

kale558k_data = {
        "MMStar": 0.343,
        "BLINK": 0.363,
        "RealWorldQA": 0.439,
        "OCRBench": 218,
        "MME perception": 925,
        "MME reasoning": 220,
        "MMMU validation": 0.318,
        "POPE": 79.8,
    }
kale558k_data['Average'] = compute_average(kale558k_data)

midtune_data = {
    "MMStar": 0.355,
    "BLINK": 0.361,
    "RealWorldQA": 0.459,
    "OCRBench": 263,
    "MME perception": 1012,
    "MME reasoning": 248,
    "MMMU validation": 0.3,
    "POPE": 85.0,
}
midtune_data['Average'] = compute_average(midtune_data)

output_dir = './plots/Midtune_blip558k'
os.makedirs(output_dir, exist_ok=True)
# 创建 8 张图，每张图对应一个列表
for i, (key, values) in enumerate(data_plot.items()):
    plt.figure(i)  # 每次绘制新的一张图
    x_values = np.arange(1, len(values)+1, 1)  # 创建横坐标
    plt.plot(x_values,values, label='Midtune_blip558k')  # 绘制折线图
    
    baseline = baseline_data[key]
    midtune = kale558k_data[key]
    midtune_base = midtune_data[key]
    plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    # plt.axhline(y=midtune, color='b', linestyle='--', label='kale558k')
    plt.axhline(y=midtune_base, color='g', linestyle='--', label='midtune_base')
    
    plt.title(f'Line Plot for {key}')  # 设置图标题
    plt.xlabel('Epochs')  # 设置 x 轴标签
    plt.ylabel('Value')  # 设置 y 轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    save_path = os.path.join(output_dir, f'{key}_plot.png')  # 使用键作为文件名的一部分
    plt.savefig(save_path, dpi=300)  # 保存图像，dpi 控制分辨率（300 为高质量）
    
    # 清理当前图，以便绘制下一张图
    plt.close()  # 关闭当前图像，防止图像堆积在内存中