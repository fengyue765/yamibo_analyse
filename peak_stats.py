import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# === 配置 ===
INPUT_FILE = "result/sentiment/sentiment_daily_global.csv"
OUTPUT_DIR = "result/sentiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 核心算法参数 ===
WINDOW_SIZE = 12        # 滑动窗口大小 (12周约等于一个季度)
Z_SCORE_THRESHOLD = 2.0 # 阈值 (大于2倍标准差视为尖峰)

def aggregate_weekly(df_daily):
    """日度 -> 周度聚合"""
    df = df_daily.copy()
    # 还原绝对量以便加权
    df['pos_count'] = df['total_comments'] * df['pos_ratio']
    df['neg_count'] = df['total_comments'] * df['neg_ratio']
    
    # 重采样 (W-MON: 每周一结束)
    df_agg = df.resample('W-MON').sum(numeric_only=True)
    df_agg = df_agg[df_agg['total_comments'] > 0].copy()
    
    # 计算极性
    df_agg['polarity_index'] = (df_agg['pos_count'] - df_agg['neg_count']) / df_agg['total_comments']
    df_agg.reset_index(inplace=True)
    return df_agg

def format_week_range(date_obj):
    """将单个日期转换为 'YYYY-MM-DD ~ YYYY-MM-DD' 格式"""
    end_date = date_obj
    start_date = end_date - pd.Timedelta(days=6)
    return f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

def detect_activity_peaks(df):
    """使用滑动窗口 Z-Score 算法检测热度尖峰"""
    df = df.copy()
    
    # 1. 计算移动平均 (Baseline) 和 标准差
    df['rolling_mean'] = df['total_comments'].rolling(window=WINDOW_SIZE, center=False, min_periods=1).mean()
    df['rolling_std'] = df['total_comments'].rolling(window=WINDOW_SIZE, center=False, min_periods=1).std()
    
    # 2. 填充 std 为 0 的情况
    df['rolling_std'] = df['rolling_std'].replace(0, 1) 
    
    # 3. 计算 Z-Score
    df['z_score'] = (df['total_comments'] - df['rolling_mean']) / df['rolling_std']
    
    # 4. 筛选尖峰
    peaks = df[df['z_score'] > Z_SCORE_THRESHOLD].copy()
    
    # 5. 格式化周范围
    peaks['week_range'] = peaks['date'].apply(format_week_range)
    
    return df, peaks

def plot_activity_chart(df_weekly, peaks):
    """绘制热度趋势图"""
    fig = go.Figure()

    # 1. 灰色柱子：实际热度
    fig.add_trace(go.Bar(
        x=df_weekly['date'], y=df_weekly['total_comments'],
        name="周热度", marker_color='rgba(180, 180, 180, 0.4)'
    ))

    # 2. 蓝色线：移动平均线
    fig.add_trace(go.Scatter(
        x=df_weekly['date'], y=df_weekly['rolling_mean'],
        name="季度平均线 (Baseline)",
        line=dict(color='#3498DB', width=2, dash='dot')
    ))

    # 3. 红色散点：爆发点
    if not peaks.empty:
        fig.add_trace(go.Scatter(
            x=peaks['date'], y=peaks['total_comments'],
            mode='markers', name="热度爆发点",
            marker=dict(color='red', size=8, symbol='star'),
            text=[f"周: {w}<br>强度: {z:.1f}x" for w, z in zip(peaks['week_range'], peaks['z_score'])],
            hoverinfo='text+y'
        ))

    fig.update_layout(
        title="<b>社区讨论热度动态监测 (Z-Score)</b>",
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="评论数"
    )
    
    output_html = os.path.join(OUTPUT_DIR, "activity_dynamic_peaks.html")
    fig.write_html(output_html)
    print(f"[Activity] 图表已生成: {output_html}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 文件 {INPUT_FILE} 不存在")
        return

    print("[Activity] 正在处理讨论热度...")
    df_daily = pd.read_csv(INPUT_FILE, parse_dates=['date'])
    df_daily.set_index('date', inplace=True)
    
    df_weekly = aggregate_weekly(df_daily)
    
    # 检测
    df_analyzed, peaks = detect_activity_peaks(df_weekly)
    
    # 导出 CSV
    output_cols = ['week_range', 'total_comments', 'rolling_mean', 'z_score', 'polarity_index']
    peaks_out = peaks[output_cols].copy()
    peaks_out['explosion_ratio'] = (peaks_out['total_comments'] / peaks_out['rolling_mean']).round(2)
    peaks_out = peaks_out.sort_values('z_score', ascending=False)
    
    csv_path = os.path.join(OUTPUT_DIR, "weekly_activity_peaks.csv")
    peaks_out.to_csv(csv_path, index=False)
    print(f"[Activity] 发现 {len(peaks)} 个爆发周，数据已保存至 {csv_path}")
    
    # 绘图
    plot_activity_chart(df_analyzed, peaks)

if __name__ == "__main__":
    main()