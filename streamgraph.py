import pandas as pd
import plotly.graph_objects as go
import os

# ==========================================
#              用户自定义配置区
# ==========================================

# 1. 在这里定义你想在图中展示哪些类，并给它们起个好听的名字
# 格式：TopicID: "自定义名称"
# 提示：您可以先运行一次脚本，看控制台输出的前20个大类，然后回来修改这里
TOPIC_MAPPING = {
    # 64: "",
    69: "美少女战士",
    87: "明日方舟",
    # 92: "银魂",
    # 100: "死神",
    # 106: "大剑",
    # 121: "SIMOUN",
    # 124: "犬夜叉",
    # 126: "",
    # 150: "海贼王",
    151: "摇曳百合",
    # 160: "",
    # 165: "",
    # 174: "",
    # 175: "名侦探柯南",
    # 179: "零～濡鸦之巫女～",
    # 181: "高达系列",
    # 186: "妖精的尾巴",
    # 188: "",
    # 202: "火影忍者",
    # 205: "",
    # 212: "",
    # 213: "",
    # 219: "",
    # 223: "",
    226: "食灵-零-",
    237: "中V",
    # 244: "EVA",
    # 247: "",
    # 246: "赛马娘"
}

# 2. 如果有些类你想合并展示 (比如 Topic 5 和 6 都属于"游戏")
# 格式：新名称: [TopicID列表]
MERGED_TOPICS = {
    "舰队collection": [30, 90],
    "魔法少女奈叶": [34, 56, 67, 155, 231],
    "VOCALOID": [52, 70],
    "Bangdream老五团": [74, 78],
    "舞hime": [35, 58, 185],
    "圣母在上": [66, 96, 221],
    "火焰纹章 风花雪月": [172],
    "孤独摇滚": [132],
    "惊爆草莓": [134],
    "LoveLive缪": [77, 102, 108, 127, 190],
    "Mygo_Mujica": [111, 241],
    "结城友奈是勇者": [119],
    "天才麻将少女": [82, 125, 242],
    "东方project": [37, 130],
    "AKB48": [75, 137, 191, 199],
    "魔禁系列": [139],
    "Fate系列": [142, 164],
    "魔法少女小圆": [59, 149],
    "魔法老师NGM": [156],
    "少女歌剧": [71, 161],
    "RWBY": [99, 171],
    # "进击的巨人": [16, 192],
    "强袭魔女": [84, 209],
    "LoveLive水": [105, 214],
    "吹响吧！上低音号": [109, 229],
    "神无月的巫女": [42],
    "战姬绝唱": [45],
    "轻音少女": [50],
    "水星的魔女": [62],
    "少女与战车": [76],
    "冰雪奇缘": [80],
    "LoveLive虹": [81],
    # "超次元游戏": [195],
    "转生王女与天才千金的魔法革命": [200]
}

# 3. 基础配置
CLUSTER_FILE = "result/nlp/clusters/topic_clusters.csv" 
HISTORY_FILE = "result/nlp/weekly_top20_general_all.csv"
OUTPUT_FILE = "result/viz/topic_evolution_streamgraph.html"
# 【阈值配置】
# 绘图时，如果某一周的平滑热度(纵坐标)低于此值，直接显示为0
PLOT_THRESHOLD = 100  

# ==========================================
#              核心逻辑实现
# ==========================================

def load_data():
    """加载并预处理数据"""
    if not os.path.exists(CLUSTER_FILE) or not os.path.exists(HISTORY_FILE):
        print(f"错误：找不到输入文件。\n检查: {CLUSTER_FILE}\n检查: {HISTORY_FILE}")
        return None, None

    print("正在加载聚类定义...")
    df_clusters = pd.read_csv(CLUSTER_FILE)
    word_to_topic = {}
    
    for _, row in df_clusters.iterrows():
        kw_str = row['keywords'] if 'keywords' in row else row['all_words']
        words = str(kw_str).replace('"', '').replace(',', ' ').split()
        tid = row['cluster_id']
        for w in words:
            word_to_topic[w.strip()] = tid

    print("正在加载历史热度数据...")
    df_hist = pd.read_csv(HISTORY_FILE)
    
    if 'week_range' in df_hist.columns:
        df_hist['date'] = df_hist['week_range'].apply(lambda x: str(x).split('~')[0].strip())
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    
    df_hist['topic_id'] = df_hist['word'].map(word_to_topic)
    df_clean = df_hist.dropna(subset=['topic_id'])
    
    return df_clean, df_clusters

def process_evolution(df_clean):
    """计算话题演变"""
    print("正在聚合数据...")
    
    df_evo = df_clean.groupby(['date', 'topic_id'])['count'].sum().reset_index()
    
    def get_topic_name(tid):
        for name, ids in MERGED_TOPICS.items():
            if tid in ids:
                return name
        if tid in TOPIC_MAPPING:
            return TOPIC_MAPPING[tid]
        return None 

    df_evo['topic_name'] = df_evo['topic_id'].apply(get_topic_name)
    df_final = df_evo.dropna(subset=['topic_name'])
    
    if df_final.empty:
        print("警告：没有数据被选中。请检查映射配置。")
        return pd.DataFrame()

    # 按新名字汇总
    df_final = df_final.groupby(['date', 'topic_name'])['count'].sum().reset_index()
    
    # 透视表
    df_pivot = df_final.pivot(index='date', columns='topic_name', values='count').fillna(0)

    # === 1. 原来的动态区间平均作图法 (8周滑动平均) ===
    print("执行 8周 滑动平均平滑...")
    df_smooth = df_pivot.rolling(window=8, min_periods=1).mean()
    
    # === 2. 绘图阈值截断 ===
    # 按照您的要求：如果纵坐标(即平滑后的值)不超过 PLOT_THRESHOLD，直接化成0
    print(f"执行阈值截断：平滑热度 < {PLOT_THRESHOLD} 设为 0...")
    df_plot_ready = df_smooth.applymap(lambda x: x if x >= PLOT_THRESHOLD else 0)

    # 还原为长表
    df_plot = df_plot_ready.reset_index().melt(id_vars='date', var_name='Topic', value_name='Heat')
    
    # 过滤掉全为0的数据点（Plotly绘图时，两点之间如果是0会自动连线到底部，形成断层效果）
    # 但为了保证时间轴连续，我们最好保留0值，或者让Plotly处理堆叠
    # 这里的过滤主要是为了减少输出文件体积，去掉那些从头到尾全是0的话题
    
    return df_plot

def draw_streamgraph(df_plot):
    """绘制河流图"""
    if df_plot.empty:
        return

    # 检查有效话题：如果一个话题在截断后所有值都是0，就彻底不画它
    topic_max_vals = df_plot.groupby('Topic')['Heat'].max()
    valid_topics = topic_max_vals[topic_max_vals > 0].index.tolist()
    
    df_plot = df_plot[df_plot['Topic'].isin(valid_topics)]
    
    if df_plot.empty:
        print(f"错误：所有话题的热度峰值均未超过阈值 {PLOT_THRESHOLD}，无法绘图。")
        return

    print(f"正在绘制河流图，包含 {len(valid_topics)} 个有效话题...")
    
    fig = go.Figure()

    # 按总热度排序图层
    topic_sums = df_plot.groupby('Topic')['Heat'].sum().sort_values(ascending=False)
    sorted_topics = topic_sums.index.tolist()

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
    ]

    for i, topic in enumerate(sorted_topics):
        data = df_plot[df_plot['Topic'] == topic]
        
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['Heat'],
            mode='none',
            name=topic,
            stackgroup='one',
            fillcolor=colors[i % len(colors)],
            hoverinfo='x+y+name'
        ))

    # 图例放在下方
    fig.update_layout(
        title=f"百合会论坛话题热度演变史",
        xaxis_title="年份",
        yaxis_title="热度 (8周滑动平均)",
        hovermode="x unified",
        template="plotly_white",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=150)
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fig.write_html(OUTPUT_FILE)
    print(f"图表已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    df_clean, df_clusters = load_data()
    if df_clean is not None:
        df_plot = process_evolution(df_clean)
        draw_streamgraph(df_plot)