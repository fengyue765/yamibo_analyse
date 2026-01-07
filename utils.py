import os
import re
import time
import logging

def sanitize_filename(filename):
    filename = re.sub(r"[\\/:*?\"<>|\s]+", "_", filename)
    filename = re.sub(r"_+", "_", filename)
    return filename.strip("._")[:80]

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def get_date_str(datetime_text):
    """
    提取日期字符串（YYYY-MM-DD），用于分目录保存。
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})", datetime_text)
    if m:
        return m.group(1)
    else:
        return time.strftime("%Y-%m-%d", time.localtime())

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_logger(name='yamibo_spider', log_file='yamibo_spider.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        # 文件输出
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger