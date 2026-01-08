import pandas as pd
import os

# === 配置路径 ===
# 请根据您实际的文件位置修改
NLP_DIR = "result/nlp"            # 存放 weekly_top20_general_all.csv 的目录
SENTIMENT_DIR = "result/sentiment" # 存放 peaks 文件的目录
OUTPUT_DIR = "result/merged_analysis" # 结果输出目录

# 文件名
FILE_KEYWORDS = "weekly_top20_general_all.csv"
FILE_SENTIMENT_PEAKS = "sentiment_peaks_list.csv"
FILE_ACTIVITY_PEAKS = "weekly_activity_peaks.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_week_range(date_str):
    """
    清洗日期格式，去除所有空格，确保可以匹配。
    例如: "2022-01-01 ~ 2022-01-07" -> "2022-01-01~2022-01-07"
    """
    if pd.isna(date_str):
        return ""
    return str(date_str).replace(" ", "")

def load_and_process_keywords(filepath):
    """
    读取热词表，并将每周的 Top20 词合并成一个字符串
    返回: 一个以 week_range 为索引的 Series
    """
    print(f"正在读取热词文件: {filepath}")
    if not os.path.exists(filepath):
        print(f"错误: 找不到文件 {filepath}")
        return None

    df = pd.read_csv(filepath)
    
    # 1. 清洗 Key
    df['join_key'] = df['week_range'].apply(normalize_week_range)
    
    # 2. 聚合逻辑
    # 将同一周的所有词和词频拼接成: "素子(54) OTL(50) GL(46)..."
    def combine_words(group):
        # 按 count 倒序排列确保顺序正确
        group = group.sort_values('count', ascending=False)
        # 拼接
        words_str = " ".join([f"{str(w)}({str(c)})" for w, c in zip(group['word'], group['count'])])
        return words_str

    print("正在聚合每周热词...")
    # Groupby 并应用拼接函数
    series_keywords = df.groupby('join_key').apply(combine_words)
    return series_keywords

def merge_and_save(peak_file_name, keywords_series, output_name):
    """
    将处理好的 keywords 合并到尖峰文件中
    """
    input_path = os.path.join(SENTIMENT_DIR, peak_file_name)
    if not os.path.exists(input_path):
        print(f"跳过: 找不到 {input_path}")
        return

    print(f"正在处理尖峰表: {peak_file_name}")
    df_peaks = pd.read_csv(input_path)
    
    # 1. 生成用于匹配的 Key (去除空格)
    df_peaks['join_key'] = df_peaks['week_range'].apply(normalize_week_range)
    
    # 2. 合并 (Left Join，保留所有尖峰行)
    # map 效率比 merge 高，适合这种 1对1 的映射
    df_peaks['top20_keywords'] = df_peaks['join_key'].map(keywords_series)
    
    # 3. 填充未找到的情况
    df_peaks['top20_keywords'] = df_peaks['top20_keywords'].fillna("无数据")
    
    # 4. 删除临时的 join_key 列
    df_peaks.drop(columns=['join_key'], inplace=True)
    
    # 5. 保存
    output_path = os.path.join(OUTPUT_DIR, output_name)
    df_peaks.to_csv(output_path, index=False)
    print(f"  -> 已生成增强版报表: {output_path}")

def main():
    # 1. 准备热词数据
    keywords_path = os.path.join(NLP_DIR, FILE_KEYWORDS)
    keywords_series = load_and_process_keywords(keywords_path)
    
    if keywords_series is None:
        return

    # 2. 处理情感尖峰 (Sentiment Peaks)
    merge_and_save(
        FILE_SENTIMENT_PEAKS, 
        keywords_series, 
        "sentiment_peaks_with_keywords.csv"
    )

    # 3. 处理热度尖峰 (Activity Peaks)
    merge_and_save(
        FILE_ACTIVITY_PEAKS, 
        keywords_series, 
        "activity_peaks_with_keywords.csv"
    )

    print("\n全部完成！请查看 results/merged_analysis 目录。")

if __name__ == "__main__":
    main()