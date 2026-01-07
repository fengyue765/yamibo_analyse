import os
import re
import time
import csv
import json
import threading
import requests
import random
from bs4 import BeautifulSoup
import concurrent.futures
from urllib.parse import urljoin
from utils import (
    sanitize_filename,
    get_timestamp,
    ensure_dir,
    get_date_str,
    get_logger
)
from downloader import download_images_and_attachments
from proxy_manager import LocalProxyReader
from config import ALWAYS_RESCAN_TOPN

# 忽略SSL警告
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def robust_get(session, url, spider_instance=None, retry=5):
    """
    增强版 GET：支持随机 User-Agent 和 自动切换代理
    :param spider_instance: 传入 YamiboSpider 实例以访问 proxy_manager
    """
    # 过滤无效 URL
    if not url or url.startswith("javascript"):
        raise ValueError(f"Invalid URL: {url}")

    # 准备 User-Agent 列表 (简单的伪装)
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
    ]

    for i in range(retry):
        # 1. 随机休眠 (模拟人类)
        time.sleep(random.uniform(1.0, 3.0))
        
        # 2. 获取代理 (如果有)
        # 尝试获取代理
        if spider_instance and getattr(spider_instance, 'use_proxy', False):
            # 确保这里访问的是 proxy_reader
            if hasattr(spider_instance, 'proxy_reader'):
                current_proxy = spider_instance.proxy_reader.get_proxy()
        
        # 3. 构造请求参数
        kwargs = {
            "timeout": 15,
            "verify": False, # 免费代理大多不支持证书验证，必须 False
            "headers": {"User-Agent": random.choice(user_agents)}
        }
        if current_proxy:
            kwargs["proxies"] = current_proxy

        try:
            # 发起请求
            # print(f"尝试请求 (代理: {current_proxy['http'] if current_proxy else '直连'})...")
            resp = session.get(url, **kwargs)
            
            # 成功
            if resp.status_code == 200:
                return resp
            
            # 处理 403/429/444 等封禁状态
            if resp.status_code in [403, 429, 444, 451, 503]:
                print(f"[反爬] IP 被拒 (Status {resp.status_code})，切换代理重试...")
                if current_proxy and spider_instance:
                    spider_instance.proxy_manager.remove_proxy(current_proxy)
                continue # 进入下一次循环，换新代理

        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout, 
                requests.exceptions.ProxyError,
                requests.exceptions.ChunkedEncodingError) as e:
                
            if i == retry - 1:
                raise RuntimeError(f"失败: {url} | Err: {e}")
            # === 切换代理 ===
            if spider_instance and hasattr(spider_instance, 'proxy_reader'):
                # 重新从文件读一个新的
                current_proxy = spider_instance.proxy_reader.get_proxy()
        
        except Exception as e:
            print(f"未知错误: {e}")

    raise RuntimeError(f"多次重试后依然获取失败：{url}")

class YamiboSpider:
    BASE_URL = "https://bbs.yamibo.com/"
    THREAD_LIST_URL_PATTERN = re.compile(r"^forum-(\d+)-\d+\.html$")
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://bbs.yamibo.com/forum.php", # 伪造来源
        "Connection": "keep-alive" # 保持连接
    }
    PAGE_SLEEP = 3.0 

    def __init__(self, save_root="data", logger=None, session=None, scan_forum_threads=True):
        self.session = session or requests.Session()
        self.session.headers.update(self.HEADERS)
        self.save_root = save_root
        self.finished_file = os.path.join(self.save_root, "finished_threads.txt")
        self.finished_lock = threading.Lock()
        self.finished = set()
        self.logger = logger or get_logger()
        self.scan_forum_threads = scan_forum_threads

        # 加载已完成记录
        if os.path.exists(self.finished_file):
            with open(self.finished_file, "r", encoding="utf-8") as f:
                for line in f:
                    self.finished.add(line.strip())
        
        # 1. 初始化读取器 (变量名必须是 proxy_reader，与 robust_get 里一致)
        self.proxy_reader = LocalProxyReader()
        
        # 2. 开关标记
        self.use_proxy = True 

    def mark_finished(self, thread_url):
        """线程安全追加已爬取的主题URL到记录文件。"""
        with self.finished_lock:
            if thread_url not in self.finished:
                with open(self.finished_file, "a", encoding="utf-8") as f:
                    f.write(thread_url.strip() + '\n')
                self.finished.add(thread_url.strip())

    def run_concurrent(self, max_workers=5):
        """
        并发执行爬取任务。
        已优化：使用分批提交 (Batch Submission) 防止内存溢出和卡死。
        """
        forum_ids = self.get_forum_ids()
        self.logger.info(f"检测到板块ID: {forum_ids}")
        
        thread_url_to_forumid = {}
        all_threads = []
        thread_urls_seen = set()
        
        # 1. 收集任务
        for fid in forum_ids:
            thread_urls = self.crawl_forum_threads(fid)
            for url in thread_urls:
                # 严格过滤无效链接
                if not url or "javascript" in url or url == "#": continue
                
                if url not in thread_urls_seen:
                    thread_url_to_forumid[url] = fid
                    all_threads.append(url)
                    thread_urls_seen.add(url)
            self.logger.info(f"板块{fid} 主题总数：{len(thread_urls)}")

        # 2. 过滤已完成任务
        pending_tasks = []
        skipped_count = 0
        for url in all_threads:
            if url in self.finished:
                skipped_count += 1
            else:
                pending_tasks.append(url)

        self.logger.info(f"总任务: {len(all_threads)} | 已完成: {skipped_count} | 待抓取: {len(pending_tasks)}")
        if not pending_tasks:
            self.logger.info("所有任务均已完成。")
            return

        random.shuffle(pending_tasks)

        # 3. 分批执行 (解决卡死的关键)
        BATCH_SIZE = 100  # 每次只提交1000个任务到线程池
        total_pending = len(pending_tasks)
        
        # 包装函数：捕获异常防止线程退出
        def crawl_wrapper(url):
            fid = thread_url_to_forumid.get(url)
            try:
                self.crawl_single_thread(url, forum_id=fid)
            except Exception as e:
                self.logger.error(f"抓取失败 {url}: {e}")

        self.logger.info(f"开始分批执行，每批 {BATCH_SIZE} 个...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 按批次切分任务
            for i in range(0, total_pending, BATCH_SIZE):
                batch = pending_tasks[i : i + BATCH_SIZE]
                current_batch_num = i // BATCH_SIZE + 1
                total_batches = (total_pending + BATCH_SIZE - 1) // BATCH_SIZE
                
                self.logger.info(f"正在提交第 {current_batch_num}/{total_batches} 批任务 ({len(batch)}个)...")
                
                # 提交当前批次
                futures = {executor.submit(crawl_wrapper, url): url for url in batch}
                
                # 等待当前批次完成，防止内存堆积
                for future in concurrent.futures.as_completed(futures):
                    pass
                
                # 打印总进度
                current_progress = min(i + BATCH_SIZE, total_pending)
                self.logger.info(f"进度: {current_progress}/{total_pending} ({(current_progress/total_pending)*100:.1f}%) 已提交完成")

        self.logger.info("所有批次执行完毕。")

    def get_forum_ids(self):
        try:
            resp = robust_get(self.session, self.BASE_URL, spider_instance=self)
            soup = BeautifulSoup(resp.text, "html.parser")
            forum_ids = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                match = self.THREAD_LIST_URL_PATTERN.search(href)
                if match:
                    forum_ids.add(int(match.group(1)))
            return list(forum_ids)
        except Exception as e:
            self.logger.error(f"获取板块列表失败: {e}")
            return []

    def crawl_forum_threads(self, forum_id, max_pages=None):
        forum_thread_list_file = os.path.join(self.save_root, f"forum_{forum_id}_threads.txt")
        forum_scan_page_file = os.path.join(self.save_root, f"forum_{forum_id}_scanpage.txt")
        thread_urls = []
        scanned_pages = set()

        # 加载本地缓存
        if os.path.exists(forum_thread_list_file):
            with open(forum_thread_list_file, "r", encoding="utf-8") as f:
                thread_urls = [line.strip() for line in f if line.strip()]
        if os.path.exists(forum_scan_page_file):
            with open(forum_scan_page_file, "r", encoding="utf-8") as f:
                scanned_pages = set(int(x) for x in f.read().splitlines() if x.isdigit())

        if not self.scan_forum_threads:
            self.logger.info(f"[只用本地主题URL] 跳过 {forum_thread_list_file} 的扫描，仅采集已存在URL。")
            return thread_urls

        # 获取最大页数
        try:
            list_url = urljoin(self.BASE_URL, f"forum-{forum_id}-1.html")
            resp = robust_get(self.session, list_url, spider_instance=self)
            soup = BeautifulSoup(resp.text, "html.parser")
            max_page = 1
            last_a = soup.find("a", class_="last")
            if last_a and last_a.has_attr('href'):
                m = re.search(r"-([0-9]+)\.html", last_a['href'])
                if m:
                    max_page = int(m.group(1))
            else:
                all_pgs = soup.select("div.pg a")
                for x in all_pgs:
                    try:
                        pg = int(x.text.strip())
                        if pg > max_page: max_page = pg
                    except: pass
        except Exception as e:
            self.logger.error(f"无法获取板块{forum_id}页数: {e}")
            return thread_urls

        if max_pages:
            max_page = min(max_page, max_pages)

        always_rescan_pages = set(range(1, ALWAYS_RESCAN_TOPN + 1))
        cur_urls_set = set(thread_urls)

        for page in range(1, max_page + 1):
            if page in scanned_pages and page not in always_rescan_pages:
                continue
            
            list_url = urljoin(self.BASE_URL, f"forum-{forum_id}-{page}.html")
            self.logger.info(f"板块{forum_id} 第{page}页: {list_url}")
            
            try:
                resp = robust_get(self.session, list_url, spider_instance=self)
            except Exception as e:
                self.logger.warning(f"板块{forum_id} 第{page}页请求失败：{e}")
                continue
                
            soup = BeautifulSoup(resp.text, "html.parser")
            thread_links = soup.select("a.xst")
            
            if not thread_links and page > 1:
                self.logger.warning(f"板块{forum_id} 第{page}页未发现主题，可能已空。")
                if page not in always_rescan_pages:
                    with open(forum_scan_page_file, "a", encoding="utf-8") as f:
                        f.write(f"{page}\n")
                continue
                
            new_count = 0
            with open(forum_thread_list_file, "a", encoding="utf-8") as f:
                for link in thread_links:
                    href = link["href"]
                    # 严格过滤
                    if not href or "javascript" in href or href == "#":
                        continue
                        
                    if href.startswith("forum.php?mod=viewthread"):
                        full_url = urljoin(self.BASE_URL, href)
                    elif href.startswith("thread-"):
                        full_url = urljoin(self.BASE_URL, href)
                    else:
                        continue
                        
                    if full_url not in cur_urls_set:
                        thread_urls.append(full_url)
                        cur_urls_set.add(full_url)
                        f.write(full_url + "\n")
                        new_count += 1
                        
            if page not in always_rescan_pages:
                with open(forum_scan_page_file, "a", encoding="utf-8") as f:
                    f.write(f"{page}\n")
            
            self.logger.info(f"板块{forum_id} 第{page}页新增{new_count}个主题，总计{len(thread_urls)}")
            time.sleep(self.PAGE_SLEEP)

        # 重新读取以确保顺序
        if os.path.exists(forum_thread_list_file):
            with open(forum_thread_list_file, "r", encoding="utf-8") as f:
                thread_urls = [line.strip() for line in f if line.strip()]
        return thread_urls

    def crawl_single_thread(self, thread_url, forum_id=None):
        # 1. 前置过滤无效URL
        if not thread_url or "javascript" in thread_url:
            return
            
        # 2. 内存检查，静默跳过（减少日志刷屏）
        if thread_url in self.finished:
            return

        try:
            resp = robust_get(self.session, thread_url, spider_instance=self)
        except Exception as e:
            self.logger.error(f"无法访问帖子 {thread_url}: {e}")
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("span", id="thread_subject") or soup.find("title")
        raw_title = title_tag.text.strip() if title_tag else "未知标题"
        title = sanitize_filename(raw_title)

        first_post = soup.find("div", id=re.compile(r"^post_\d+$"))
        post_time = ""
        if first_post:
            time_tag = first_post.find("em", id=re.compile(r"^authorposton"))
            post_time = time_tag.text.strip().replace("发表于", "") if time_tag else ""
        
        date_str = get_date_str(post_time)
        timestamp = get_timestamp()
        
        # 路径生成
        save_dir = os.path.join(self.save_root, date_str, f"{title}_{timestamp}")
        meta_path = os.path.join(save_dir, "meta.json")
        
        # 二次检查本地文件（防止重复抓取）
        if os.path.exists(meta_path):
            self.mark_finished(thread_url)
            return

        ensure_dir(save_dir)
        replies = []
        page = 1
        floor_num = 0
        
        # 翻页循环
        while True:
            posts = soup.find_all("div", id=re.compile(r"^post_\d+$"))
            for post in posts:
                floor_num += 1
                try:
                    reply_data = self.extract_post_data(post, save_dir, floor_num)
                    replies.append(reply_data)
                except Exception as e:
                    self.logger.warning(f"无法解析楼层: {e}")
            
            # 查找下一页
            next_page = soup.find("a", class_="nxt")
            if not next_page:
                break
                
            href = next_page.get("href")
            # 检查下一页链接有效性
            if not href or "javascript" in href.lower() or href.strip() == "#":
                break
                
            next_url = urljoin(self.BASE_URL, href)
            
            # 防止死循环：下一页就是当前页
            if next_url == thread_url or next_url == resp.url:
                break
            
            try:
                resp = robust_get(self.session, next_url, spider_instance=self)
                soup = BeautifulSoup(resp.text, "html.parser")
                page += 1
                time.sleep(self.PAGE_SLEEP)
            except Exception as e:
                self.logger.warning(f"翻页失败 {next_url}: {e}")
                break

        # 保存数据
        meta = {
            "thread_url": thread_url,
            "forum_id": forum_id,
            "title": raw_title,
            "num_replies": len(replies),
            "saved_at": timestamp,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        replies_csv_path = os.path.join(save_dir, "replies.csv")
        with open(replies_csv_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["楼层号", "用户名", "发表时间", "内容", "本地图片列表", "本地附件列表", "forum_id"]
            )
            for r in replies:
                writer.writerow([
                    r['floor'],
                    r['username'],
                    r['time'],
                    r['content'],
                    ";".join(r['images'] or []),
                    ";".join(r['attachments'] or []),
                    forum_id
                ])
                
        self.mark_finished(thread_url)
        # 恢复了此处的日志输出
        self.logger.info(f"[DONE] ⬇️ 帖子‘{raw_title}’已保存({len(replies)}楼)：{save_dir}")

    def extract_post_data(self, post, save_dir, floor_no):
        user_tag = post.find("a", class_="xw1")
        username = user_tag.text.strip() if user_tag else "匿名"
        time_tag = post.find("em", id=re.compile(r"^authorposton"))
        post_time = time_tag.text.strip().replace("发表于", "") if time_tag else ""
        content_div = post.find("td", class_="t_f")
        content_raw = content_div.decode_contents() if content_div else ""
        
        # 调用下载器下载图片
        images, attachments, content_text = download_images_and_attachments(
            content_raw, save_dir, self.session
        )
        return {
            "floor": f"{floor_no}#",
            "username": username,
            "time": post_time,
            "content": content_text,
            "images": images,
            "attachments": attachments
        }