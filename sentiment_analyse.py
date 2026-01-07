import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime
import gc

# === 配置 ===
DATA_ROOT = "data"
OUTPUT_DIR = "result/sentiment"
LIT_FORUM_ID = '49'  # 文学区ID
BATCH_SIZE = 64      # 4060Ti 16G 显存很大，可以设大一点加速

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        return 0  # 使用第一块 GPU
    return -1     # 使用 CPU

def load_data_generator(target_type='all'):
    """
    数据生成器：流式读取数据，避免内存溢出
    target_type: 'all' (全站) 或 'lit' (文学区)
    """
    date_dirs = sorted([d for d in os.listdir(DATA_ROOT) if len(d) == 10])
    
    for date_str in date_dirs:
        date_path = os.path.join(DATA_ROOT, date_str)
        try:
            post_dirs = os.listdir(date_path)
        except:
            continue
            
        for post_name in post_dirs:
            post_path = os.path.join(date_path, post_name)
            meta_path = os.path.join(post_path, "meta.json")
            csv_path = os.path.join(post_path, "replies_cleaned.csv")
            
            # 1. 检查板块 ID (如果是文学区模式)
            is_lit = False
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        if str(json.load(f).get('forum_id')) == LIT_FORUM_ID:
                            is_lit = True
                except: pass
            
            # 如果只跑文学区且当前不是文学区，跳过
            if target_type == 'lit' and not is_lit:
                continue
            
            # 2. 读取回复内容
            if os.path.exists(csv_path):
                try:
                    try:
                        df = pd.read_csv(csv_path, usecols=['内容'], encoding='utf-8', on_bad_lines='skip')
                    except:
                        df = pd.read_csv(csv_path, usecols=['内容'], encoding='gb18030', on_bad_lines='skip')
                    
                    contents = df['内容'].dropna().astype(str).tolist()
                    
                    # === 关键策略 ===
                    # 如果是文学区，跳过第一条（通常是小说正文），只看读者评论
                    # 除非只有一个楼层（只有正文没回复），那就跳过
                    if target_type == 'lit':
                        if len(contents) > 1:
                            contents = contents[1:]
                        else:
                            continue 
                            
                    for text in contents:
                        # 文本太短通常是无效情感（如“顶”、“好”），太长需要截断
                        # 模型通常限制 512 token，我们取前 300 字符足够判断情感
                        clean_text = text.strip()
                        if len(clean_text) > 2: 
                            yield {
                                "date": date_str,
                                "text": clean_text[:300] 
                            }
                except:
                    continue

def run_analysis(task_name, target_type):
    print(f"\n=== 开始任务: {task_name} (GPU加速中...) ===")
    
    # 加载模型
    # lxyuan/distilbert-base-multilingual-cased-sentiments-student
    # 这是一个轻量级、多语言支持极好的情感分析模型
    model_name = "./distilbert-base-multilingual-cased-sentiments-student"
    
    print(f"正在加载模型 {model_name} ...")
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=get_device(),
        top_k=None, # 返回所有标签的得分，方便计算加权分
        truncation=True,
        batch_size=BATCH_SIZE
    )
    
    # 结果聚合器: { "2023-01-01": { "pos": 100, "neg": 20, "score_sum": 55.5, "count": 120 } }
    daily_stats = {}
    
    # 创建生成器
    data_gen = load_data_generator(target_type)
    
    # 使用 pipe 批量推理
    # 这里的 tqdm total 设为 None 因为不知道具体有多少条
    buffer_dates = []
    buffer_texts = []
    
    # 映射标签到分数
    # positive -> 1, neutral -> 0.5, negative -> 0
    
    for item in tqdm(data_gen, desc="情感推理中", unit="msg"):
        buffer_dates.append(item['date'])
        buffer_texts.append(item['text'])
        
        # 凑够一个 Batch 再处理，或者由 pipeline 内部处理
        # 这里的 pipeline(iterator) 模式会自动处理 batching
        pass

    # 由于 pipeline 只能接受 iterable，我们需要重新构造一下以配合 pipeline 的高效模式
    # 上面的循环只是为了演示逻辑，下面是真正高效的写法：
    
    def generator_wrapper():
        for item in load_data_generator(target_type):
            yield item['text']

    # 我们需要另外记录日期，因为 pipeline 只返回结果不返回原始输入
    # 这在流式处理中比较麻烦。
    # 为了准确对应日期，我们还是手动分批处理比较稳妥。
    
    dataset_iterator = load_data_generator(target_type)
    
    current_batch_texts = []
    current_batch_dates = []
    
    for item in tqdm(dataset_iterator, desc=f"Analyzing {task_name}"):
        current_batch_texts.append(item['text'])
        current_batch_dates.append(item['date'])
        
        if len(current_batch_texts) >= BATCH_SIZE:
            # 推理
            results = classifier(current_batch_texts)
            
            # 统计结果
            for date_str, res_list in zip(current_batch_dates, results):
                if date_str not in daily_stats:
                    daily_stats[date_str] = {'pos': 0, 'neg': 0, 'neu': 0, 'total': 0, 'score_sum': 0.0}
                
                # res_list 是 [{'label': 'positive', 'score': 0.9}, {'label': 'negative', 'score': 0.05}...]
                # 我们需要找到 score 最高的 label，或者计算加权分
                # 为了简单，直接取 top label
                
                # 转换 pipeline 输出格式 (lxyuan模型输出的是列表)
                # 格式通常是 sorted list of dicts
                top_res = sorted(res_list, key=lambda x: x['score'], reverse=True)[0]
                label = top_res['label']
                score = top_res['score'] # 置信度
                
                # 计算一个归一化的情感分 (0~1)
                # Positive = 1, Neutral = 0.5, Negative = 0
                normalized_score = 0.5
                if label == 'positive':
                    daily_stats[date_str]['pos'] += 1
                    normalized_score = 1.0 * score # 置信度越高越接近1
                elif label == 'negative':
                    daily_stats[date_str]['neg'] += 1
                    normalized_score = 0.0 + (1-score) # 置信度越高越接近0，置信度低则接近0.5
                else:
                    daily_stats[date_str]['neu'] += 1
                    normalized_score = 0.5
                
                daily_stats[date_str]['total'] += 1
                daily_stats[date_str]['score_sum'] += normalized_score

            # 清空 Buffer
            current_batch_texts = []
            current_batch_dates = []
    
    # 处理剩余的
    if current_batch_texts:
        results = classifier(current_batch_texts)
        for date_str, res_list in zip(current_batch_dates, results):
            if date_str not in daily_stats:
                daily_stats[date_str] = {'pos': 0, 'neg': 0, 'neu': 0, 'total': 0, 'score_sum': 0.0}
            
            top_res = sorted(res_list, key=lambda x: x['score'], reverse=True)[0]
            label = top_res['label']
            
            if label == 'positive': daily_stats[date_str]['pos'] += 1
            elif label == 'negative': daily_stats[date_str]['neg'] += 1
            else: daily_stats[date_str]['neu'] += 1
            
            daily_stats[date_str]['total'] += 1
            # 简化的分数累加
            val = 1.0 if label == 'positive' else (0.0 if label == 'negative' else 0.5)
            daily_stats[date_str]['score_sum'] += val

    # === 生成报表 ===
    print("正在聚合统计结果...")
    rows = []
    for date_str, stat in daily_stats.items():
        total = stat['total']
        if total > 0:
            avg_score = stat['score_sum'] / total
            pos_ratio = stat['pos'] / total
            neg_ratio = stat['neg'] / total
            
            # 情感极性 = (积极数 - 消极数) / 总数
            # 范围 -1 到 1
            polarity = (stat['pos'] - stat['neg']) / total
            
            rows.append({
                "date": date_str,
                "total_comments": total,
                "avg_sentiment_score": round(avg_score, 4), # 0(消极)~1(积极)
                "pos_ratio": round(pos_ratio, 4),
                "neg_ratio": round(neg_ratio, 4),
                "polarity_index": round(polarity, 4) # 市场情绪指标
            })
            
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    filename = f"sentiment_daily_{task_name}.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"完成！文件已保存: {output_path}")
    
    # 释放显存
    del classifier
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. 跑文学区 (优先，因为数据量小，测试快)
    # run_analysis("literature", "lit")
    
    # 2. 跑全站
    run_analysis("global", "all")