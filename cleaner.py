import pandas as pd
from bs4 import BeautifulSoup
from opencc import OpenCC
import re
import os
import shutil
import warnings
import logging
import sys
import threading
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

# 忽略警告
warnings.filterwarnings("ignore")

# === 配置日志 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cleaner.log", encoding='utf-8', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# === 核心清洗类 ===
class TextCleaner:
    def __init__(self):
        self.cc = OpenCC('t2s')
        self.edit_status = re.compile(r'(\[\s*(本帖最后由|Last edited by).*?(编辑|at\s*[\d:]+)\s*\])', re.I | re.S)
        self.garbage = [
            re.compile(r'\(\s*\d+(\.\d+)?\s*[KMGT]B\s*,\s*下载次数.*?\)', re.I),
            re.compile(r'(下载附件|保存到相册|点击文件名下载附件|上传附件)(?=\s|$)', re.I),
            re.compile(r'\n\s*\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}(\s*上传)?\s*(?=\n)'),
            re.compile(r'^\s*[\w\-\(\)\.]+\.(jpg|jpeg|png|gif|rar|zip)\s*$', re.I | re.M)
        ]
        self.hide = re.compile(r'(\*\*\*\*\s*本内容被作者隐藏\s*\*\*\*\*|Post by .*? at .*? Has ban post|本帖隐藏的内容需要积分高于)', re.I)
        self.smilies = [re.compile(r'yamiboqe\d+', re.I), re.compile(r'll\d+', re.I)]

    def clean_html(self, html):
        if pd.isna(html) or str(html).strip() == "": return ""
        s = str(html)
        s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
        s = re.sub(r'</(div|p|li|h\d|tr)>', '\n', s, flags=re.I)
        
        try:
            soup = BeautifulSoup(s, "html.parser")
            for t in soup.find_all(['script', 'style', 'iframe', 'img']): t.decompose()
            for c in ['pstatus', 'quote', 'attach_nopermission', 'tip', 'alert_error', 'ignore_js_op', 'y']:
                for t in soup.find_all(class_=c): t.decompose()
            text = soup.get_text(separator='', strip=True)
        except:
            text = re.sub(r'<[^>]+>', '', s)

        text = self.edit_status.sub('', text)
        for p in self.garbage: text = p.sub('', text)
        text = self.hide.sub('', text)
        for p in self.smilies: text = p.sub('', text)
        
        text = self.cc.convert(text)
        return re.sub(r'\n\s*\n\s*\n+', '\n\n', text).strip()

cleaner_instance = None

def init_worker():
    global cleaner_instance
    cleaner_instance = TextCleaner()

def worker_clean_csv(file_path):
    global cleaner_instance
    try:
        output_path = os.path.join(os.path.dirname(file_path), "replies_cleaned.csv")
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        except:
            df = pd.read_csv(file_path, encoding='gb18030', on_bad_lines='skip', engine='python')

        if '内容' not in df.columns: return False

        df['内容_cleaned'] = df['内容'].apply(cleaner_instance.clean_html)
        
        if '楼层号' in df.columns: df['楼层号'] = df['楼层号'].astype(str).str.replace('#', '', regex=False).str.strip()
        if '发表时间' in df.columns: df['发表时间'] = df['发表时间'].astype(str).str.strip()
        if '用户名' in df.columns: df['用户名'] = df['用户名'].astype(str).apply(lambda x: cleaner_instance.cc.convert(str(x)))

        export_cols = ['楼层号', '用户名', '发表时间', '内容_cleaned', '本地图片列表', '本地附件列表', 'forum_id']
        final_cols = [c for c in export_cols if c in df.columns]
        
        if final_cols:
            df_clean = df[final_cols].rename(columns={'内容_cleaned': '内容'})
            df_clean.dropna(how='all', inplace=True)
            df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        logger.error(f"Failed: {file_path} | {e}")
        return False

class GlobalManager:
    def __init__(self, root_dir):
        self.root = root_dir
        self.cc = OpenCC('t2s')
        self.regex_date = re.compile(r'(?:_更新下载)?_(\d{8}_\d{6})\s*$')
        self.regex_unix = re.compile(r'_(\d{10,13})\s*$')

    def normalize_name(self, name):
        ts = 0
        clean_name = name
        match_date = self.regex_date.search(name)
        match_unix = self.regex_unix.search(name)
        
        if match_date:
            ts_str = match_date.group(1).replace('_', '')
            ts = int(ts_str) 
            clean_name = name[:match_date.start()]
        elif match_unix:
            ts = int(match_unix.group(1))
            clean_name = name[:match_unix.start()]
            
        final_name = self.cc.convert(clean_name).strip()
        return final_name, ts

    def step3_clean_content_streaming(self):
        """流式处理：修复版 - 使用回调函数实时更新进度"""
        print("[Step 3] 准备流式多进程清洗 CSV...")
        
        if not os.path.exists(self.root):
            print("根目录不存在")
            return

        # WSL 优化：限制并发数
        max_workers = max(1, os.cpu_count() - 1)
        # 如果觉得卡顿依然严重，可以手动改为 max_workers = 4
        
        print(f"    启动 {max_workers} 个进程，正在扫描并实时处理...")
        
        level1_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        # 线程锁，用于多线程安全地更新进度条
        pbar_lock = threading.Lock()
        
        # 统计数据
        stats = {"submitted": 0, "completed": 0}
        
        # 两个进度条
        # 1. 扫描进度
        scan_pbar = tqdm(total=len(level1_dirs), desc="扫描目录", unit="dir", position=0)
        # 2. 清洗进度 (初始 total=0，后续动态增加)
        clean_pbar = tqdm(total=0, desc="清洗进度", unit="file", position=1)

        def on_task_done(future):
            """回调函数：当一个文件清洗完成后触发"""
            with pbar_lock:
                clean_pbar.update(1)
                stats["completed"] += 1
                
                # 可选：检查是否有异常（记录日志）
                if future.exception():
                    logger.error(f"Task failed: {future.exception()}")

        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            pending_futures = []
            
            for l1 in level1_dirs:
                l1_path = os.path.join(self.root, l1)
                try:
                    level2_dirs = os.listdir(l1_path)
                    
                    for l2 in level2_dirs:
                        target_file = os.path.join(l1_path, l2, "replies.csv")
                        
                        # 提交任务
                        future = executor.submit(worker_clean_csv, target_file)
                        
                        # === 关键修正：绑定回调函数 ===
                        # 只要任务做完，就会自动调用 on_task_done 更新进度条
                        future.add_done_callback(on_task_done)
                        
                        pending_futures.append(future)
                        stats["submitted"] += 1
                        
                        # 动态更新进度条的总量
                        with pbar_lock:
                            clean_pbar.total = stats["submitted"]
                            clean_pbar.refresh()

                        # 背压控制 (Backpressure)
                        # 如果积压的任务太多 (>20000)，暂停提交，防止内存爆炸
                        # 只需要检查 pending_futures 列表的大小
                        if len(pending_futures) > 20000:
                            # 清理已完成的引用
                            pending_futures = [f for f in pending_futures if not f.done()]
                            # 如果清理后依然很多，稍微阻塞一下
                            if len(pending_futures) > 15000:
                                import time
                                time.sleep(0.1) 

                except OSError:
                    pass
                
                scan_pbar.update(1)
            
            scan_pbar.close()
            print("\n扫描完成，等待剩余任务结束...")
            
            # 不需要再用 as_completed 循环了，ProcessPoolExecutor 退出时会默认等待所有任务完成
            # 我们只需要保持进度条显示直到结束
            
        clean_pbar.close()
        print("\n全部任务完成！")

    def run(self):
        print("请选择操作模式:")
        print("1. 执行完整流程 (索引 -> 去重 -> 清洗)")
        print("2. 仅执行数据清洗 (跳过目录重组)")
        choice = input("请输入选项 (1/2): ").strip()

        if choice == '1':
            self.step1_step2_reorganize()
            self.step3_clean_content_streaming() # 使用新的流式处理
        else:
            print(">>> 跳过 Step 1 & 2，直接进入 Step 3...")
            self.step3_clean_content_streaming()

    # ... (step1_step2_reorganize 代码保持不变) ...
    def step1_step2_reorganize(self):
        print(f"[Step 1] 正在扫描 {self.root} (二级目录模式) 建立索引...")
        title_map = {}
        scan_count = 0
        if not os.path.exists(self.root):
            print("根目录不存在")
            return
        level1_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        for l1 in level1_dirs:
            l1_path = os.path.join(self.root, l1)
            try:
                level2_dirs = [d for d in os.listdir(l1_path) if os.path.isdir(os.path.join(l1_path, d))]
            except OSError:
                continue
            for l2 in level2_dirs:
                current_post_path = os.path.join(l1_path, l2)
                clean_name, ts = self.normalize_name(l2)
                if ts == 0:
                    try:
                        ts = int(os.path.getmtime(current_post_path))
                    except:
                        ts = 0
                if clean_name not in title_map:
                    title_map[clean_name] = []
                title_map[clean_name].append({
                    "ts": ts,
                    "path": current_post_path,
                    "original_name": l2
                })
                scan_count += 1
                if scan_count % 5000 == 0:
                    print(f"    -> 已索引 {scan_count} 个目录...", end="\r")
        print(f"\n索引完成，共发现 {len(title_map)} 个唯一标题组 (原始目录数: {scan_count})")
        print("[Step 2] 执行全局去重与重命名...")
        rename_ops = []
        delete_ops = []
        for title, items in title_map.items():
            items.sort(key=lambda x: x['ts'], reverse=True)
            keep_item = items[0]
            if len(items) > 1:
                for remove_item in items[1:]:
                    delete_ops.append(remove_item['path'])
                    logger.info(f"[标记删除] {remove_item['original_name']} (保留: {keep_item['original_name']})")
            current_path = keep_item['path']
            parent_dir = os.path.dirname(current_path)
            target_path = os.path.join(parent_dir, title)
            if os.path.abspath(current_path) != os.path.abspath(target_path):
                 rename_ops.append((current_path, target_path))
        if delete_ops:
            print(f"    正在删除 {len(delete_ops)} 个重复帖子...")
            for p in tqdm(delete_ops, unit="dir"):
                try:
                    if os.path.exists(p):
                        trash = p + "_TRASH"
                        os.rename(p, trash)
                        shutil.rmtree(trash)
                except Exception as e:
                    logger.error(f"删除失败 {p}: {e}")
        if rename_ops:
            print(f"    正在重命名 {len(rename_ops)} 个帖子...")
            for src, dst in tqdm(rename_ops, unit="dir"):
                try:
                    if os.path.exists(src):
                        if os.path.exists(dst):
                            continue
                        os.rename(src, dst)
                except Exception as e:
                    logger.error(f"重命名失败 {src} -> {dst}: {e}")

if __name__ == "__main__":
    DATA_ROOT = "data"  # 请根据实际情况修改
    if not os.path.exists(DATA_ROOT):
        print(f"目录不存在: {DATA_ROOT}")
    else:
        mgr = GlobalManager(DATA_ROOT)
        mgr.run()