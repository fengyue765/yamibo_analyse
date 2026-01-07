import subprocess
import time
import os
import sys
import signal
from config import NUM_WORKERS, SPIDER_SCRIPT, TIMEOUT_SECONDS

# 全局变量存储所有子进程，方便信号处理函数访问
processes = []
proxy_service = None  # <--- 新增：定义全局变量
PROXY_SERVICE_SCRIPT = "proxy_manager.py"

def start_worker(index):
    """启动单个爬虫进程"""
    if not os.path.exists("heartbeat"):
        os.makedirs("heartbeat")
    # 为每个进程指定独立的心跳文件（虽然你随机了batch，但独立心跳能让你知道具体死的是哪个）
    hb_file = f"heartbeat/spider_heartbeat_{index}.txt"
    
    # 初始化心跳
    with open(hb_file, "w") as f:
        f.write(str(time.time()))
    
    # 传递环境变量，让spider知道往哪个文件写心跳
    env = os.environ.copy()
    env["SPIDER_HEARTBEAT_FILE"] = hb_file
    
    print(f"[监控] 启动 Worker-{index} (PID待定)...")
    
    # 关键点：使用 preexec_fn=os.setsid (Linux/WSL特有)
    # 这会将子进程放入新的进程组，但在简单的 Ctrl+C 场景下，
    # 我们主要靠主进程捕获信号后手动 kill。
    proc = subprocess.Popen(
        [sys.executable, SPIDER_SCRIPT],
        env=env,
        # preexec_fn=os.setsid # 如果你需要完全解耦可以加这个，但这里我们要手动管理，暂不加
    )
    return {
        "proc": proc,
        "index": index,
        "hb_file": hb_file
    }

def kill_worker(worker_info):
    """强制关闭单个进程"""
    proc = worker_info["proc"]
    try:
        if proc.poll() is None: # 只有进程还活着才杀
            print(f"[监控] 正在终止 Worker-{worker_info['index']} (PID: {proc.pid})...")
            os.kill(proc.pid, signal.SIGKILL) # 简单粗暴，确保杀死
            proc.wait() # 等待资源回收，防止僵尸进程
    except Exception as e:
        print(f"关闭 Worker-{worker_info['index']} 失败: {e}")

def signal_handler(signum, frame):
    """捕获 Ctrl+C 信号，优雅退出"""
    global proxy_service
    
    print("\n\n[系统] 接收到中断信号，正在清理所有子进程...")
    
    # 1. 暴力清理爬虫进程
    for p in processes:
        if p and p["proc"]:
            try:
                # 发送 SIGKILL 确保必死
                if p["proc"].poll() is None:
                    os.kill(p["proc"].pid, signal.SIGKILL)
            except:
                pass
            
    # 2. 暴力清理代理服务
    if proxy_service:
        print("[监控] 正在关闭代理服务...")
        try:
            if proxy_service.poll() is None:
                os.kill(proxy_service.pid, signal.SIGKILL) # 强制杀死
                proxy_service.wait() # 避免僵尸进程
        except Exception as e:
            print(f"清理代理服务出错: {e}")
        
    print("所有进程已清理，监控退出。")
    sys.exit(0)

def main():
    global proxy_service # <--- 声明使用全局变量
    print("[监控] 正在启动全局代理服务进程...")
    # 使用 Popen 后台运行，不阻塞
    proxy_service = subprocess.Popen([sys.executable, PROXY_SERVICE_SCRIPT])
    
    # 等待几秒，确保它生成了初始的 proxies_verified.json 文件
    print("等待代理池初始化 (20秒)...")
    time.sleep(20)

    # 注册信号处理器：当按下 Ctrl+C 时执行 signal_handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[监控] 正在启动 {NUM_WORKERS} 个并行爬虫...")

    # 1. 启动所有 Worker
    for i in range(NUM_WORKERS):
        worker = start_worker(i)
        processes.append(worker)
        time.sleep(30) # 错峰启动

    print("所有 Worker 已启动，开始循环监控...")

    while True:
        time.sleep(10)
        
        for i in range(NUM_WORKERS):
            worker = processes[i]
            proc = worker["proc"]
            hb_file = worker["hb_file"]
            
            # A. 检查进程是否意外退出
            if proc.poll() is not None:
                print(f"[监控] Worker-{i} 意外退出了 (Exit Code: {proc.returncode})，正在重启...")
                time.sleep(300)
                processes[i] = start_worker(i) # 重启并更新列表
                continue
            
            # B. 检查心跳
            if not os.path.exists(hb_file):
                # 文件没了补一个
                with open(hb_file, "w") as f: f.write(str(time.time()))
                continue
                
            try:
                last_modified = os.path.getmtime(hb_file)
                idle_time = time.time() - last_modified
                
                if idle_time > TIMEOUT_SECONDS:
                    print(f"[监控] Worker-{i} 已卡死 {int(idle_time)}s，正在强制重启...")
                    kill_worker(worker)
                    processes[i] = start_worker(i)
                if proxy_service.poll() is not None:
                    print("[监控] 代理服务挂了，正在重启...")
                    proxy_service = subprocess.Popen([sys.executable, PROXY_SERVICE_SCRIPT])
                else:
                    # 正常
                    pass
            except Exception as e:
                print(f"[监控] 检查 Worker-{i} 出错: {e}")

if __name__ == "__main__":
    main()