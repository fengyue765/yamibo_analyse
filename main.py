import os
from spider import YamiboSpider
from utils import get_logger
from config import USERNAME, PASSWORD, LOGIN_URL, SCAN_FORUM_THREADS, THREADS
from login_discuz import discuz_login

def main():
    save_root = "data"
    os.makedirs(save_root, exist_ok=True)
    logger = get_logger(log_file="yamibo_spider.log")
    # --- 登陆Discuz ---
    print("[INFO] 正在登录Discuz论坛...")
    try:
        # 现在的 discuz_login 内部会自动换代理重试了
        # 如果您的 IP 被封，它会尝试用代理池里的 IP 去撞
        session = discuz_login(USERNAME, PASSWORD, LOGIN_URL)
    except Exception as e:
        print(f"FATAL ERROR: 登录彻底失败，程序退出。{e}")
        return # 或者 sys.exit(1)
    # --- 传递已登录session给爬虫 ---
    spider = YamiboSpider(save_root=save_root, logger=logger, session=session, scan_forum_threads=SCAN_FORUM_THREADS)
    spider.run_concurrent(max_workers=THREADS)

if __name__ == "__main__":
    main()