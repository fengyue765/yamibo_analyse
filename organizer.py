import os
import shutil
import pandas as pd
import logging
from datetime import datetime
import re
import random
import sys

# === 配置日志 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("organizer.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostOrganizer:
    def __init__(self, root_dir):
        self.root = root_dir
        self.fallback_dir = os.path.join(self.root, "Unknown_Date")
        
        # 目标源目录范围 (仅用于 Mode 1)
        self.target_range_start = datetime(2026, 1, 1)
        self.target_range_end = datetime(2026, 1, 6)

    def is_target_source_dir(self, dir_name):
        """检查一级目录名是否在 2026-01-01 到 2026-01-06 之间"""
        try:
            d = datetime.strptime(dir_name, "%Y-%m-%d")
            return self.target_range_start <= d <= self.target_range_end
        except ValueError:
            return False

    def get_post_date(self, csv_path, is_raw_csv=False):
        """
        读取CSV，获取1楼发表时间，返回 YYYY-MM-DD 字符串。
        :param is_raw_csv: 如果是原始 replies.csv，楼层号通常带有 '#' (如 1#)
        """
        try:
            # 尝试不同编码读取
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip', engine='python')
            except:
                df = pd.read_csv(csv_path, encoding='gb18030', on_bad_lines='skip', engine='python')
            
            # 兼容列名 (原始CSV可能是 '楼层', '时间'; 清洗后是 '楼层号', '发表时间')
            col_floor = '楼层' if '楼层' in df.columns else '楼层号'
            col_time = '时间' if '时间' in df.columns else '发表时间'

            if col_floor not in df.columns or col_time not in df.columns:
                return None

            # 转换为字符串去空格
            df[col_floor] = df[col_floor].astype(str).str.strip()
            
            # 寻找1楼
            # 逻辑：去除 '#' 和 '.0' 后等于 '1'
            def clean_floor_num(x):
                return str(x).replace('#', '').replace('.0', '').strip()

            first_floor = df[df[col_floor].apply(clean_floor_num) == '1']
            
            if first_floor.empty:
                # 备选：如果只有一行数据且看起来像主楼
                if not df.empty:
                    row = df.iloc[0]
                else:
                    return None
            else:
                row = first_floor.iloc[0]

            time_str = str(row[col_time]).strip()
            
            # 解析时间
            dt = pd.to_datetime(time_str, errors='coerce')
            
            if pd.isna(dt):
                return None
                
            return dt.strftime("%Y-%m-%d")

        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            return None

    def move_folder(self, src_path, target_parent_dir):
        """安全移动文件夹，处理重名情况"""
        folder_name = os.path.basename(src_path)
        
        # 确保目标父目录存在
        if not os.path.exists(target_parent_dir):
            os.makedirs(target_parent_dir)
            
        dest_path = os.path.join(target_parent_dir, folder_name)
        
        # 如果目标路径已存在同名文件夹，改名避让
        while os.path.exists(dest_path):
            if os.path.abspath(src_path) == os.path.abspath(dest_path):
                return
            
            logger.warning(f"目标冲突: {dest_path} 已存在，���在重命名...")
            new_name = f"{folder_name}_dup_{random.randint(1000, 9999)}"
            dest_path = os.path.join(target_parent_dir, new_name)
            
        try:
            shutil.move(src_path, dest_path)
            logger.info(f"Moved: {src_path} -> {dest_path}")
        except Exception as e:
            logger.error(f"Move Failed: {src_path} -> {dest_path} | {e}")

    def run_mode_standard(self):
        """Mode 1: 标准整理 (针对特定日期范围的目录)"""
        print(f"=== Mode 1: 整理 {self.root} 下的帖子 ===")
        print(f"扫描范围: {self.target_range_start.date()} 至 {self.target_range_end.date()}")

        try:
            top_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        except FileNotFoundError:
            print(f"错误: 根目录 {self.root} 不存在")
            return

        for parent_dir in top_dirs:
            if not self.is_target_source_dir(parent_dir):
                continue
            
            parent_path = os.path.join(self.root, parent_dir)
            print(f"正在扫描目录: {parent_dir}")
            
            post_dirs = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
            
            if not post_dirs:
                try: os.rmdir(parent_path)
                except: pass
                continue

            for post_name in post_dirs:
                post_path = os.path.join(parent_path, post_name)
                # 优先找 cleaned，没有的话逻辑上它就留在这里或者去 unknown
                # 但根据原需求，Mode 1 主要依赖 replies_cleaned.csv
                csv_path = os.path.join(post_path, "replies_cleaned.csv")
                
                target_date_str = None
                if os.path.exists(csv_path):
                    target_date_str = self.get_post_date(csv_path)
                
                if target_date_str:
                    final_target_dir = os.path.join(self.root, target_date_str)
                else:
                    final_target_dir = self.fallback_dir
                    
                self.move_folder(post_path, final_target_dir)

        # 清理空目录
        for parent_dir in top_dirs:
            if self.is_target_source_dir(parent_dir):
                p_path = os.path.join(self.root, parent_dir)
                if os.path.exists(p_path) and not os.listdir(p_path):
                    os.rmdir(p_path)

        print("Mode 1 整理完成！")

    def run_mode_unknown_fix(self):
        """Mode 2: 修复 Unknown_Date 目录下的帖子"""
        print(f"=== Mode 2: 正在扫描 {self.fallback_dir} 进行二次归档 ===")
        
        if not os.path.exists(self.fallback_dir):
            print(f"目录不存在: {self.fallback_dir}，无需处理。")
            return

        post_dirs = [d for d in os.listdir(self.fallback_dir) if os.path.isdir(os.path.join(self.fallback_dir, d))]
        print(f"发现 {len(post_dirs)} 个待处理文件夹...")

        success_count = 0
        fail_count = 0

        for post_name in post_dirs:
            post_path = os.path.join(self.fallback_dir, post_name)
            
            # 策略：优先找 replies.csv (原始文件)，因为 Mode 1 没处理好通常是因为没洗过
            # 如果有 cleaned 也可以用
            raw_csv = os.path.join(post_path, "replies.csv")
            clean_csv = os.path.join(post_path, "replies_cleaned.csv")
            
            target_date_str = None
            
            # 1. 尝试从原始 replies.csv 提取
            if os.path.exists(raw_csv):
                target_date_str = self.get_post_date(raw_csv, is_raw_csv=True)
            
            # 2. 如果失败，尝试从 replies_cleaned.csv 提取
            if not target_date_str and os.path.exists(clean_csv):
                target_date_str = self.get_post_date(clean_csv, is_raw_csv=False)

            if target_date_str:
                # 找到了有效日期，移出 Unknown_Date
                final_target_dir = os.path.join(self.root, target_date_str)
                self.move_folder(post_path, final_target_dir)
                success_count += 1
            else:
                # 依然找不到日期，留在原地
                fail_count += 1
                # logger.warning(f"依然无法解析: {post_name}")

        print(f"\nMode 2 完成: 成功迁移 {success_count} 个，剩余 {fail_count} 个仍保留在 Unknown_Date。")

    def run(self):
        print("请选择操作模式:")
        print("1. 标准整理 (扫描 2026-01-01 至 2026-01-06 的目录，基于 replies_cleaned.csv)")
        print("2. 修复 Unknown_Date (扫描 Unknown_Date 目录，尝试使用 replies.csv 进行二次归档)")
        
        choice = input("请输入选项 (1/2): ").strip()
        
        if choice == '1':
            self.run_mode_standard()
        elif choice == '2':
            self.run_mode_unknown_fix()
        else:
            print("无效选项，退出。")

if __name__ == "__main__":
    DATA_ROOT = "data"  # 修改为你的实际路径
    organizer = PostOrganizer(DATA_ROOT)
    organizer.run()