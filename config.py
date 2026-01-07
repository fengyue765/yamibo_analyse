USERNAME = "fengyue765"
PASSWORD = "xh031012"
LOGIN_URL = "https://bbs.yamibo.com/member.php?mod=logging&action=login"
# 验证码图片URL（如遇验证码需求，需补充获取方式；如无可留空）
CAPTCHA_URL = ""    # 若需要验证码，请补充实际图片地址

SCAN_FORUM_THREADS = False   # 是否需要重新扫描帖子前X页
ALWAYS_RESCAN_TOPN = 5  # 重扫前X页

THREADS = 8    # 每个进程并行线程数
NUM_WORKERS = 8     # 并行子进程数

SPIDER_SCRIPT = "main.py"  # 你的爬虫脚本名
TIMEOUT_SECONDS = 900        # 进程运行超时时间

CHECK_INTERVAL = 300  # 每 5 分钟验证一轮IP池
CHECK_NUM = 2000 # 在所有原始代理中抽取进行验证的个数