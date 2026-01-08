import os
import re
import json
import jieba
import jieba.posseg as pseg
import pandas as pd
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import multiprocessing
import csv

# === 配置区域 ===
DATA_ROOT = "data"
RESULTS_DIR = "result/nlp"
VIZ_DIR = "result/viz"
INTERMEDIATE_DIR = "result/intermediates"
STOPWORDS_FILE = "stopwords.txt"
TARGET_FORUM_ID = '49'

# 临时文件夹 (用于存放每天的中间统计结果，最后合并)
TEMP_DIR = os.path.join(INTERMEDIATE_DIR, "temp_stats")

ALLOWED_POS_PREFIX = ('n', 'eng', 'x', 'vn') 
DEFAULT_STOPWORDS = {
    'yamibo', 'yamibohk', 'lz', 'LZ', 'Lz', '楼上', '楼主', '沙发', '板凳',
    '附件', '下载', '上传', '最后', '编辑', '积分', '回复', '帖子', '点击',
    'contents', '作者', '文案', '正文', '全文', '章节', '字数', '内容',
    '链接', '图片', '权限', '评分', '引用', '发表', '注册', '登录',
    '时候', '时间', '今天', '明天', '昨天', '现在', '曾经', '未来', '永远',
    '东西', '事情', '问题', '样子', '地方', '部分', '全部', '所有', '大家',
    '可能', '虽然', '但是', '因为', '所以', '其实', '真的', '非常', '比较',
    '什么', '怎么', '哪里', '那里', '这里', '一个', '没有', '自己', '知道',
    '感觉', '觉得', '喜欢', '看到', '看着', '出来', '进去', '开始', '结束',
    '不过', '一下', '一点', '一些', '一直', '一定', '还是', '只是', '就是',
    'br', 'img', 'jpg', 'png', 'gif', 'url', 'http', 'https'
}

for d in [RESULTS_DIR, VIZ_DIR, INTERMEDIATE_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
worker_stopwords = None

def load_stopwords():
    sw = DEFAULT_STOPWORDS.copy()
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                sw.add(line.strip())
    return sw

def init_worker(stopwords_set):
    global worker_stopwords
    worker_stopwords = stopwords_set
    jieba.initialize()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'yamibo[a-zA-Z0-9]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\u4e00-\u9fa5\u3040-\u30ff\u30a0-\u30ff\uff66-\uff9fa-zA-Z]', ' ', text)
    return text.strip()

def process_day_task(args):
    """Worker: 处理单日数据"""
    date_str, root_path = args
    date_path = os.path.join(root_path, date_str)
    
    stats = {
        "general_all": Counter(),
        "general_title": Counter(),
        "literature_all": Counter(),
        "literature_title": Counter()
    }
    corpus_lines = []
    jp_pattern = re.compile(r'[\u3040-\u30ff\u30a0-\u30ff]') 
    
    try:
        post_dirs = os.listdir(date_path)
    except:
        return None

    for post_name in post_dirs:
        post_path = os.path.join(date_path, post_name)
        is_lit = False
        meta_path = os.path.join(post_path, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    if str(meta.get('forum_id')) == TARGET_FORUM_ID:
                        is_lit = True
            except: pass

        raw_title = re.sub(r'_\d{8}_\d{6}$', '', post_name)
        content_texts = []
        csv_path = os.path.join(post_path, "replies_cleaned.csv")
        if os.path.exists(csv_path):
            try:
                try:
                    df = pd.read_csv(csv_path, usecols=['内容'], encoding='utf-8', on_bad_lines='skip')
                except:
                    df = pd.read_csv(csv_path, usecols=['内容'], encoding='gb18030', on_bad_lines='skip')
                content_texts = df['内容'].dropna().astype(str).tolist()
            except: pass
            
        # 处理标题
        clean_t = clean_text(raw_title)
        if clean_t:
            words = pseg.cut(clean_t)
            t_tokens = []
            for w, flag in words:
                if len(w) < 2 or w in worker_stopwords: continue
                if flag.startswith(ALLOWED_POS_PREFIX) or \
                   ((flag == 'x' or flag == 'eng') and (jp_pattern.search(w) or w.isascii())):
                    t_tokens.append(w)
                    stats["general_title"][w] += 1
                    stats["general_all"][w] += 1
                    if is_lit:
                        stats["literature_title"][w] += 1
                        stats["literature_all"][w] += 1
            if t_tokens: corpus_lines.append(" ".join(t_tokens))

        # 处理正文
        for raw_c in content_texts:
            clean_c = clean_text(raw_c)
            if not clean_c: continue
            words = pseg.cut(clean_c)
            c_tokens = []
            for w, flag in words:
                if len(w) < 2 or w in worker_stopwords: continue
                if flag.startswith(ALLOWED_POS_PREFIX) or \
                   ((flag == 'x' or flag == 'eng') and (jp_pattern.search(w) or w.isascii())):
                    c_tokens.append(w)
                    stats["general_all"][w] += 1
                    if is_lit:
                        stats["literature_all"][w] += 1
            if c_tokens: corpus_lines.append(" ".join(c_tokens))
                        
    return (date_str, stats, corpus_lines)

class NLPManagerSafe:
    def run(self):
        print("=== 启动 NLP 分析引擎 (内存安全版) ===")
        print("策略: 边算边存，避免 20 年数据撑爆内存。")
        
        if not os.path.exists(DATA_ROOT):
            print("错误: 数据目录不存在")
            return
            
        all_dates = sorted([d for d in os.listdir(DATA_ROOT) if len(d) == 10 and d.count('-') == 2])
        stopwords = load_stopwords()
        
        # 准备文件句柄：同时写 4 个 CSV，避免把数据存内存
        # 格式: date, word, count
        file_handles = {}
        csv_writers = {}
        tasks_keys = ["general_all", "general_title", "literature_all", "literature_title"]
        
        for key in tasks_keys:
            f_path = os.path.join(TEMP_DIR, f"temp_{key}.csv")
            f = open(f_path, 'w', encoding='utf-8', newline='')
            writer = csv.writer(f)
            writer.writerow(['date', 'word', 'count']) # Header
            file_handles[key] = f
            csv_writers[key] = writer
            
        corpus_file = os.path.join(INTERMEDIATE_DIR, "corpus_full.txt")
        open(corpus_file, 'w', encoding='utf-8').close()

        print(f"Step 1: 多进程处理 {len(all_dates)} 天数据 (流式写入硬盘)...")
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(stopwords,)) as executor:
            tasks = [(d, DATA_ROOT) for d in all_dates]
            
            with open(corpus_file, 'a', encoding='utf-8') as f_corpus:
                for res in tqdm(executor.map(process_day_task, tasks, chunksize=5), total=len(tasks)):
                    if not res: continue
                    d, stats_map, lines = res
                    
                    # === 立即写入硬盘，释放内存 ===
                    for key in tasks_keys:
                        # 优化：每天只写入 Top 500 词频，避免写入几十万个长尾词导致 CSV 膨胀
                        # 如果硬盘够大，可以去掉 most_common，写入全部
                        for w, c in stats_map[key].most_common(500):
                            csv_writers[key].writerow([d, w, c])
                    
                    if lines: f_corpus.write("\n".join(lines) + "\n")
        
        # 关闭文件句柄
        for f in file_handles.values():
            f.close()
            
        print("\nStep 2-4: 读取临时文件并生成最终报表...")
        # 逐个任务处理，处理完一个释放一个 DataFrame，节省内存
        for key in tasks_keys:
            self.process_stats_file(key)

        print("\nStep 5: 训练 Word2Vec...")
        self.train_word2vec(corpus_file)

    def process_stats_file(self, task_key):
        """读取扁平的 CSV，生成各种榜单"""
        input_csv = os.path.join(TEMP_DIR, f"temp_{task_key}.csv")
        if not os.path.exists(input_csv): return
        
        print(f"  正在处理任务: {task_key} ...")
        
        # 1. 读取数据 (指定 dtype 优化内存)
        try:
            df = pd.read_csv(input_csv, parse_dates=['date'])
        except Exception as e:
            print(f"    读取失败: {e}")
            return
            
        if df.empty: return

        # === A. 生成 Bar Chart Race 累积数据 ===
        # 透视
        print("    - 生成累积宽表 (Pivot)...")
        # 过滤：只取全时段总频次较高的词进行透视，防止列数爆炸
        top_words_overall = df.groupby('word')['count'].sum().nlargest(300).index
        df_filtered = df[df['word'].isin(top_words_overall)]
        
        df_wide = df_filtered.pivot_table(index='date', columns='word', values='count', fill_value=0)
        idx = pd.date_range(df_wide.index.min(), df_wide.index.max())
        df_wide = df_wide.reindex(idx, fill_value=0)
        
        df_cumsum = df_wide.cumsum()
        
        # 保存 Top 100
        last_row = df_cumsum.iloc[-1]
        top_cols = last_row.nlargest(100).index
        
        out_race = os.path.join(VIZ_DIR, f"cumulative_{task_key}.csv")
        df_cumsum[top_cols].to_csv(out_race)
        print(f"      -> 已保存: {out_race}")

        # === B. 生成年度 Top 20 ===
        print("    - 生成年度榜单...")
        df['year'] = df['date'].dt.year
        df_yearly = df.groupby(['year', 'word'])['count'].sum().reset_index()
        df_yearly_top = df_yearly.sort_values(['year', 'count'], ascending=[True, False]).groupby('year').head(20)
        
        out_year = os.path.join(RESULTS_DIR, f"yearly_top20_{task_key}.csv")
        df_yearly_top.to_csv(out_year, index=False)

        # === C. 生成周度 Top 20 ===
        print("    - 生成周度榜单...")
        # Grouper by week (W-MON)
        df_weekly = df.groupby([pd.Grouper(key='date', freq='W-MON'), 'word'])['count'].sum().reset_index()
        df_weekly_top = df_weekly.sort_values(['date', 'count'], ascending=[True, False]).groupby('date').head(20)
        
        # 格式化周日期范围
        def format_week(d):
            start = d - pd.Timedelta(days=6)
            return f"{start.strftime('%Y-%m-%d')}~{d.strftime('%Y-%m-%d')}"
            
        df_weekly_top['week_range'] = df_weekly_top['date'].apply(format_week)
        out_week = os.path.join(RESULTS_DIR, f"weekly_top20_{task_key}.csv")
        # 调整列顺序
        cols = ['week_range', 'word', 'count']
        df_weekly_top[cols].to_csv(out_week, index=False)

        # === D. ��成历史总榜 ===
        print("    - 生成历史总榜...")
        df_total = df.groupby('word')['count'].sum().nlargest(500).reset_index()
        out_total = os.path.join(RESULTS_DIR, f"top500_{task_key}.csv")
        df_total.to_csv(out_total, index=False)
        
        # 主动释放内存
        del df, df_wide, df_cumsum, df_filtered
        import gc
        gc.collect()

    def train_word2vec(self, corpus_path):
        # 保持原样
        sentences = LineSentence(corpus_path)
        model = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=10, workers=multiprocessing.cpu_count(), epochs=5, sg=1)
        model.save(os.path.join(RESULTS_DIR, "yamibo_word2vec.model"))
        print(f"模型已保存至 {RESULTS_DIR}")

if __name__ == "__main__":
    manager = NLPManagerSafe()
    manager.run()