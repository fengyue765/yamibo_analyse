import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 配置 ===
DATA_ROOT = "data"
OUTPUT_FILE = "forum_advanced_stats.html"

# 板块映射表
FORUM_MAP = {
    '16': '管理版',
    '370': '论坛指南',
    '5': '动漫区',
    '33': '海域区',
    '13': '贴图区',
    '49': '文学区',
    '44': '游戏区',
    '379': '影视区'
}

def read_single_meta(meta_path):
    """读取单个 json 文件的辅助函数"""
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            fid = str(meta.get('forum_id', 'Other'))
            replies = meta.get('num_replies', 0)
            
            # 简单清洗
            if not isinstance(replies, (int, float)):
                replies = 0
                
            return {
                "fid": fid,
                "replies": int(replies),
                "count": 1
            }
    except:
        return None

def load_data_fast(root_path):
    """多线程并发读取数据"""
    if not os.path.exists(root_path):
        print(f"错误: 目录 {root_path} 不存在")
        return pd.DataFrame()

    # 1. 扫描所有待处理的 meta.json 路径
    print("Step 1: 扫描文件列表...")
    tasks = []
    
    # 获取日期目录
    date_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    for date_str in tqdm(date_dirs, unit="day", desc="扫描目录"):
        try:
            # 验证日期格式
            current_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
            
        day_path = os.path.join(root_path, date_str)
        try:
            post_dirs = os.listdir(day_path)
            for p in post_dirs:
                meta_path = os.path.join(day_path, p, "meta.json")
                # 暂时只存路径和日期对象，不读取内容
                tasks.append((current_date, meta_path))
        except OSError:
            continue
            
    print(f"共发现 {len(tasks)} 个帖子，准备并发读取...")

    # 2. 多线程读取内容
    results = []
    # I/O 密集型任务，线程数可以设大一点，例如 CPU核心数 * 2 或 固定 16/32
    max_workers = min(32, os.cpu_count() * 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_date = {executor.submit(read_single_meta, t[1]): t[0] for t in tasks}
        
        # 处理结果
        for future in tqdm(as_completed(future_to_date), total=len(tasks), unit="file", desc="读取元数据"):
            res = future.result()
            if res:
                res['date'] = future_to_date[future] # 把日期补回去
                results.append(res)

    return pd.DataFrame(results)

def get_forum_name(fid):
    """获取板块名称，如果没有映射则返回 ID"""
    return FORUM_MAP.get(fid, f'其他板块({fid})')

def process_time_series(df, freq):
    """
    按指定频率聚合数据
    freq: 'D' (日), 'ME' (月), 'YE' (年)
    """
    # 按日期重采样并求和
    # 需要先设日期为索引
    df_idx = df.set_index('date')
    resampled = df_idx.resample(freq).sum().reset_index()
    return resampled

def plot_advanced(df):
    if df.empty:
        print("没有数据。")
        return

    print("Step 2: 正在聚合数据...")
    
    # 1. 预处理
    df['forum_name'] = df['fid'].apply(get_forum_name)
    
    # 2. 生成不同粒度的数据集
    # 我们需要为每个板块生成 3 份数据 (日/月/年)
    
    # 获取所有板块列表
    unique_forums = sorted(df['forum_name'].unique())
    # 把“其他板块”放到最后，把重要板块放前
    priority_forums = list(FORUM_MAP.values())
    sorted_forums = [f for f in priority_forums if f in unique_forums] + [f for f in unique_forums if f not in priority_forums]
    
    # 添加“全站总计”
    all_targets = ['Total (全站)'] + sorted_forums

    print("Step 3: 生成图表对象 (这可能需要几秒钟)...")

    # 创建双坐标轴图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 用于存储菜单按钮
    buttons = []
    
    # 记录当前添加了多少条 Trace，用于控制可见性
    # 结构：每次循环添加 2 条 Trace (发帖数, 回复数)
    trace_index = 0
    total_traces_per_view = 2 

    # 循环顺序：板块 -> 时间粒度
    # 这样下拉菜单的顺序比较自然
    
    time_scales = [
        ('按日统计', 'D', 'bar'),   # 每日数据波动大，用 Bar 可能看发帖量更清楚，或者 Line 也可以
        ('按月统计', 'ME', 'bar'),  # 月度
        ('按年统计', 'YE', 'bar')   # 年度
    ]

    for target in tqdm(all_targets, desc="生成绘图层"):
        # 筛选当前板块数据
        if target == 'Total (全站)':
            current_df = df # 全部数据
        else:
            current_df = df[df['forum_name'] == target]
        
        # 如果该板块没数据，跳过
        if current_df.empty:
            continue

        # 仅保留需要的列以加速 group
        base_df = current_df[['date', 'count', 'replies']]

        for label_suffix, freq, plot_type in time_scales:
            # 聚合计算
            agg_df = process_time_series(base_df, freq)
            
            # 决定是否可见：默认只显示 "Total (全站) - 按日"
            is_visible = (target == 'Total (全站)' and freq == 'D')
            
            # --- Trace A: 发帖量 (左轴) ---
            # 使用 Bar 或 Scatter (Fill) 来表示数量
            if freq == 'D':
                # 日线太密集，用线图填充比较好看，柱状图会太细
                fig.add_trace(
                    go.Scatter(
                        x=agg_df['date'], y=agg_df['count'],
                        name="发帖数量",
                        mode='lines',
                        line=dict(width=1, color='#636EFA'),
                        fill='tozeroy', # 填充面积
                        visible=is_visible
                    ),
                    secondary_y=False
                )
            else:
                # 月/年用柱状图
                fig.add_trace(
                    go.Bar(
                        x=agg_df['date'], y=agg_df['count'],
                        name="发帖数量",
                        marker_color='#636EFA',
                        opacity=0.6,
                        visible=is_visible
                    ),
                    secondary_y=False
                )

            # --- Trace B: 回复量 (右轴) ---
            # 要求：实线
            fig.add_trace(
                go.Scatter(
                    x=agg_df['date'], y=agg_df['replies'],
                    name="总回复/楼层",
                    mode='lines', # 实线
                    line=dict(width=2.5, color='#EF553B'), # 橙红色
                    visible=is_visible
                ),
                secondary_y=True
            )
            
            # --- 创建按钮 ---
            # 计算 visibility 数组
            # 这个数组长度必须等于总 Trace 数，目前还不知道总数，所以不能静态创建
            # Plotly 的机制允许我们在 Python 端只定义逻辑，最后生成 JSON
            # 我们需要构建一个 mask
            
            menu_label = f"{target} - {label_suffix}"
            
            # 这是一个暂存的按钮信息，后面会统一修正 visibility 数组长度
            buttons.append({
                "label": menu_label,
                "index": trace_index # 记录这组数据的起始索引
            })
            
            trace_index += 2

    # 修正按钮的 visibility 逻辑
    final_buttons = []
    total_traces = trace_index
    
    for btn in buttons:
        # 创建全 False 数组
        vis_array = [False] * total_traces
        # 激活当前视角的 2 条线
        idx = btn['index']
        vis_array[idx] = True     # 发帖量
        vis_array[idx+1] = True   # 回复量
        
        final_buttons.append(dict(
            label=btn['label'],
            method="update",
            args=[
                {"visible": vis_array},
                {"title": f"论坛趋势分析: {btn['label']}"}
            ]
        ))

    # 布局设置
    fig.update_layout(
        title="论坛趋势分析 (默认: 全站日线)",
        updatemenus=[{
            "buttons": final_buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }],
        template="plotly_white",
        hovermode="x unified",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 设置坐标轴标题
    fig.update_yaxes(title_text="<b>发帖数量</b> (柱/面积)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="<b>总楼层/回复数</b> (实线)", secondary_y=True, showgrid=True)

    fig.write_html(OUTPUT_FILE)
    print(f"\n图表已生成: {os.path.abspath(OUTPUT_FILE)}")
    print("提示：请使用图表左上角的下拉菜单切换【板块】和【统计粒度（日/月/年）】。")

if __name__ == "__main__":
    # 1. 快速读取
    df = load_data_fast(DATA_ROOT)
    
    # 2. 绘图
    plot_advanced(df)