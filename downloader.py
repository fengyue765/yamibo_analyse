import os
import re
import requests
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup
from utils import ensure_dir, sanitize_filename

# 忽略SSL警告
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def download_images_and_attachments(content_html, save_dir, session):
    """
    融合版下载器：
    1. 继承您repo中的逻辑：同时抓取 <img> 和 <a> 标签链接的图片。
    2. 增强稳定性：移除硬编码代理，自动加Referer防盗链，支持 data/attachment 路径。
    """
    soup = BeautifulSoup(content_html, "html.parser")
    images_dir = os.path.join(save_dir, "images")
    attach_dir = os.path.join(save_dir, "attachments")
    ensure_dir(images_dir)
    ensure_dir(attach_dir)
    
    image_paths = []
    attach_paths = []
    base_site = "https://bbs.yamibo.com/"
    
    # 定义需要捕获的图片后缀 (来自您的 repo)
    img_exts = re.compile(r'\.(jpg|jpeg|png|gif|bmp|webp)($|\?)', re.I)
    emotion_keywords = ["smiley", "smilies", "face", "emotion", "emoji"]

    # -------------------------------------------------------------
    # 1. 收集所有待下载任务
    # -------------------------------------------------------------
    tasks = []

    # A. 提取 <img> 标签 (Discuz 特性：zoomfile > file > src)
    for img in soup.find_all("img"):
        raw_url = img.get("zoomfile") or img.get("file") or img.get("src")
        if raw_url:
            tasks.append({'tag': img, 'url': raw_url, 'type': 'img'})

    # B. 提取 <a> 标签 (您 repo 中的逻辑：链接是图片的也抓)
    for a in soup.find_all("a", href=True):
        href = a['href']
        # 如果是附件下载页
        if "attachment.php" in href:
            tasks.append({'tag': a, 'url': href, 'type': 'attachment_php'})
        # 如果链接直接指向图片文件 (repo 逻辑迁移)
        elif img_exts.search(href):
            tasks.append({'tag': a, 'url': href, 'type': 'link_img'})

    # -------------------------------------------------------------
    # 2. 执行下载
    # -------------------------------------------------------------
    # 计数器
    img_counter = 0

    for task in tasks:
        raw_url = task['url']
        tag = task['tag']
        
        # --- URL 补全 ---
        full_url = raw_url
        if not raw_url.startswith("http"):
            if raw_url.startswith("data/"): # 修复 data/attachment 无斜杠开头
                full_url = urljoin(base_site, raw_url)
            elif raw_url.startswith("/"):
                full_url = base_site.rstrip("/") + raw_url
            else:
                full_url = urljoin(base_site, raw_url)

        # --- 判定保存目录 ---
        src_low = full_url.lower()
        is_emotion = any(k in src_low for k in emotion_keywords)
        
        # 判定是否为内容图：路径含attachment、link_img类型、或者 attachment.php
        is_content = (
            "/attachment/" in src_low or 
            "/album/" in src_low or 
            task['type'] in ['link_img', 'attachment_php'] or
            "aimg_" in str(tag.get("id", ""))
        )

        if is_content:
            target_dir = attach_dir
            dest_list = attach_paths
        elif is_emotion:
            target_dir = images_dir
            dest_list = image_paths
        else:
            target_dir = images_dir
            dest_list = image_paths

        # --- 生成文件名 (img_1.jpg) ---
        img_counter += 1
        
        # 尝试从 URL 提取后缀
        ext = "jpg"
        try:
            path_obj = urlparse(full_url)
            if '.' in path_obj.path:
                candidate = path_obj.path.split('.')[-1].lower()
                if len(candidate) < 5: ext = candidate
        except: pass
        
        # 附件php通常没有后缀，先存为bin或尝试识别
        if task['type'] == 'attachment_php':
            fname = f"attachment_{img_counter}.bin"
        else:
            fname = f"img_{img_counter}.{ext}"

        local_path = os.path.join(target_dir, fname)

        # 避免文件名冲突
        while os.path.exists(local_path):
            img_counter += 1
            fname = f"img_{img_counter}.{ext}"
            local_path = os.path.join(target_dir, fname)

        # --- 下载核心 (修复 Connection refused) ---
        try:
            # 1. 构造 headers (加 Referer 防盗链)
            headers = {
                "Referer": base_site,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            # 2. 发起请求
            # 注意：不再传 proxies 参数！除非您在 spider.py 的 session 里配了。
            # 这样如果没开代理，它就会直连，不会报错。
            resp = session.get(full_url, headers=headers, timeout=20, stream=True, verify=False)
            
            if resp.status_code == 200:
                # 检查是否误下载了网页HTML (权限不足时常见)
                if 'text/html' in resp.headers.get('Content-Type', '').lower():
                    # print(f"跳过HTML: {full_url}")
                    dest_list.append(full_url)
                    continue

                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # 成功！
                rel_path = os.path.relpath(local_path, save_dir)
                dest_list.append(rel_path)
                
                # 修改 HTML 引用
                if task['type'] == 'img':
                    tag['src'] = rel_path
                    # 清理干扰属性
                    if tag.has_attr("zoomfile"): del tag["zoomfile"]
                    if tag.has_attr("file"): del tag["file"]
                else:
                    tag['href'] = rel_path
            else:
                dest_list.append(full_url)
        except Exception as e:
            # print(f"下载异常: {e}")
            dest_list.append(full_url)

    return image_paths, attach_paths, soup.decode_contents()