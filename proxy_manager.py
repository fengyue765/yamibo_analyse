import requests
import re
import threading
import time
import random
import os
import json
import signal
import sys 
from concurrent.futures import ThreadPoolExecutor
from config import CHECK_INTERVAL, CHECK_NUM

# === 配置 ===
PROXY_FILE = "proxies_verified.json"  # 共享文件路径
TARGET_URL = "https://bbs.yamibo.com/forum.php"

class ProxyService:
    def __init__(self):
        self.sources = [
            "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
            "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
            "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt",
            "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
            "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt"
        ]
        self.good_proxies = []

    def fetch_raw_proxies(self):
        """从 GitHub 拉取原始数据"""
        raw_set = set()
        print("[ProxyService] 正在拉取 GitHub 代理源...")
        for url in self.sources:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    found = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', resp.text)
                    raw_set.update(found)
            except Exception:
                pass
        print(f"[ProxyService] 获取到 {len(raw_set)} 个原始代理")
        return list(raw_set)

    def verify_proxy(self, proxy):
        """验证单个代理"""
        p = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            # 5秒超时，使用 HEAD 请求减少流量
            resp = requests.head(TARGET_URL, proxies=p, timeout=5)
            if resp.status_code < 400:
                return proxy
        except:
            pass
        return None

    def update_pool(self):
        """核心循环：拉取 -> 验证 -> 写入文件"""
        raw_proxies = self.fetch_raw_proxies()
        
        # 随机抽样验证 (避免一次验证几万个太慢)
        # 每次验证 1000 个，加上上一轮存活的代理
        check_list = list(set(raw_proxies + self.good_proxies))
        random.shuffle(check_list)
        check_list = check_list[:CHECK_NUM]

        print(f"[ProxyService] 开始验证 {len(check_list)} 个代理...")
        
        new_good_proxies = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(self.verify_proxy, p) for p in check_list]
            for f in futures:
                res = f.result()
                if res:
                    new_good_proxies.append(res)
        
        self.good_proxies = new_good_proxies
        print(f"[ProxyService] 验证完成，存活代理: {len(self.good_proxies)} 个")

        # === 写入共享文件 ===
        # 使用临时文件 + rename 原子操作，防止爬虫读取时文件为空
        temp_file = PROXY_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.good_proxies, f)
        
        if os.path.exists(PROXY_FILE):
            os.remove(PROXY_FILE)
        os.rename(temp_file, PROXY_FILE)
        print(f"[ProxyService] 已更新 {PROXY_FILE}")

    def run(self):
        while True:
            try:
                self.update_pool()
            except Exception as e:
                print(f"[ProxyService] 异常: {e}")
            
            print(f"休息 {CHECK_INTERVAL} 秒...")
            time.sleep(CHECK_INTERVAL)

# === 爬虫端使用的简易读取器 ===
# 这个类会被 spider.py 导入
class LocalProxyReader:
    def __init__(self):
        self.proxies = []
        self.last_reload = 0

    def get_proxy(self):
        # 每 30 秒重新加载一次文件
        if time.time() - self.last_reload > 30:
            self._reload()
        
        if not self.proxies:
            return None
            
        p = random.choice(self.proxies)
        return {"http": f"http://{p}", "https": f"http://{p}"}

    def _reload(self):
        if not os.path.exists(PROXY_FILE):
            return
        try:
            with open(PROXY_FILE, "r", encoding="utf-8") as f:
                self.proxies = json.load(f)
            # print(f"[Spider] 已加载 {len(self.proxies)} 个代理")
        except:
            pass

    def report_fail(self, proxy_dict):
        # 既然是只读模式，爬虫端不需要真的去删文件
        # 只要自己内存里暂时不用它就行，反正Service会定期重新验证
        pass

def signal_handler(signum, frame):
    print("\n[ProxyService] 接收到退出信号，正在停止...")
    sys.exit(0)

# 如果直接运行此脚本，则启动服务
if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service = ProxyService()
    service.run()