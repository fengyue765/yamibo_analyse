import requests
import time
from bs4 import BeautifulSoup
from proxy_manager import LocalProxyReader  # 导入我们在 proxy_manager.py 定义的读取器

def discuz_login(username, password, login_url, captcha_code=None):
    """
    Discuz 登录函数 (代理增强版)
    逻辑：
    1. 循环尝试获取代理。
    2. 使用代理 GET 登录页获取 formhash。
    3. 使用同一个代理 POST 提交账号密码。
    4. 验证是否登录成功。
    """
    
    # 初始化代理读取器
    proxy_reader = LocalProxyReader()
    
    # 最大重试次数
    max_retries = 15
    
    print(f"[登录] 开始登录流程 (最大重试 {max_retries} 次)...")

    for i in range(max_retries):
        # 每次重试都创建一个新的 Session，避免旧的脏 Cookie 干扰
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://bbs.yamibo.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        })

        # 1. 获取代理
        current_proxy = proxy_reader.get_proxy()
        proxy_str = current_proxy['http'] if current_proxy else "直连"
        
        # 准备请求参数 (超时设置短一点，快速试错)
        req_kwargs = {
            "proxies": current_proxy,
            "timeout": 10,
            "verify": False # 免费代理通常不支持 SSL 验证
        }

        try:
            print(f"[登录] 第 {i+1} 次尝试 | 代理: {proxy_str} | 获取 formhash...")
            
            # Step 1: 访问登录页，获得 formhash
            resp = session.get(login_url, **req_kwargs)
            
            if resp.status_code != 200:
                print(f"   访问登录页失败 (状态码 {resp.status_code})")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            formhash_tag = soup.find("input", attrs={"name": "formhash"})
            
            if not formhash_tag:
                print("   未找到 formhash，可能是页面加载不完整或 IP 被盾")
                continue
                
            formhash = formhash_tag["value"]
            # print(f"   获取 formhash: {formhash}")

            # Step 2: 拼接登录数据
            payload = {
                "username": username,
                "password": password,
                "formhash": formhash,
                "loginfield": "username",
                "questionid": 0,
                "answer": "",
                "referer": "https://bbs.yamibo.com/",
                "loginsubmit": "true"
            }
            if captcha_code:
                payload["seccodeverify"] = captcha_code

            # Step 3: 提交登录 (必须使用同一个 session 和 同一个 proxy)
            # print(f"   正在提交账号密码...")
            post_url = login_url
            login_resp = session.post(post_url, data=payload, **req_kwargs)
            
            # 检查是否成功
            # 增加一些判断词，yamibo登录成功通常右上角会有用户名
            success = "欢迎您" in login_resp.text or \
                      "退出" in login_resp.text or \
                      "我的帖子" in login_resp.text or \
                      f"title=\"访问我的空间\">{username}" in login_resp.text

            if success:
                print(f"[登录] 登录成功！(代理: {proxy_str})")
                
                # 【关键】虽然 Session 可以在这里返回，
                # 但 requests.Session 不会记住 proxies 参数用于后续请求。
                # 您的 spider.py 会通过 robust_get 重新给每个请求分配代理，
                # 只要 Cookie (Session ID) 有效，IP 变动通常是被允许的。
                return session
            else:
                # 尝试打印一部分内容看看到底显示啥（调试用）
                title = soup.find("title")
                title_text = title.text if title else "无标题"
                print(f"   登录提交后未检测到成功标识 (页面标题: {title_text})，尝试下一个代理...")

        except Exception as e:
            print(f"   连接错误: {e}")
            time.sleep(1) # 稍微歇一下
            continue
            
    raise RuntimeError("[登录] 所有代理尝试均失败，无法登录论坛。")