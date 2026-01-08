import pandas as pd
import os

# === 配置 ===
CSV_FILE = "stopwords.csv"
TXT_FILE = "stopwords.txt"

def merge_stopwords():
    final_stopwords = set()
    has_changes = False

    # 1. 读取现有的 TXT 文件
    if os.path.exists(TXT_FILE):
        print(f"正在读取现有文件: {TXT_FILE}")
        with open(TXT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    final_stopwords.add(word)
        print(f"  - 原有词数: {len(final_stopwords)}")
    else:
        print(f"提示: 未找到 {TXT_FILE}，将创建一个新文件。")

    # 2. 尝试读取 CSV 文件
    if os.path.exists(CSV_FILE):
        print(f"正在读取待合并文件: {CSV_FILE}")
        try:
            df = pd.read_csv(CSV_FILE)
            
            # 兼容表头：优先找 'word' 列，找不到就取第一列
            target_col = None
            if 'word' in df.columns:
                target_col = df['word']
            elif not df.empty:
                print("  - 未找到表头 'word'，尝试使用第一列作为词源。")
                target_col = df.iloc[:, 0]
            
            if target_col is not None:
                new_words = target_col.dropna().astype(str).tolist()
                count_before = len(final_stopwords)
                
                for w in new_words:
                    w = w.strip()
                    if w:
                        final_stopwords.add(w)
                
                added_count = len(final_stopwords) - count_before
                if added_count > 0:
                    print(f"  - 成功合并并新增词数: {added_count}")
                    has_changes = True
                else:
                    print("  - CSV 中的词已全部存在，无需新增。")
            else:
                print("  - CSV 文件为空或格式无法解析。")

        except Exception as e:
            print(f"读取 CSV 失败: {e}")
    else:
        print(f"提示: 未找到 {CSV_FILE}，将仅对 {TXT_FILE} 进行去重和重排序。")
        # 即使没有 CSV，我们也需要保存一次以实现去重和排序，所以视为“有变动”意图
        has_changes = True 

    # 3. 排序并保存
    # 只要集合不为空，就进行重写，确保格式统一（排序+去重）
    if final_stopwords:
        print("正在保存...")
        sorted_stopwords = sorted(list(final_stopwords))
        
        with open(TXT_FILE, 'w', encoding='utf-8') as f:
            for word in sorted_stopwords:
                f.write(word + "\n")

        print(f"处理完成！最终 {TXT_FILE} 包含 {len(sorted_stopwords)} 个停用词（已去重并排序）。")
    else:
        print("警告: 最终词表为空，未进行保存。")

if __name__ == "__main__":
    merge_stopwords()