import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# === 配置 ===
INPUT_DIR = "result/sentiment"
OUTPUT_DIR = "result/sentiment"
# 确保能找到文件，支持单文件或遍历目录逻辑
TARGET_FILE = "sentiment_daily_global.csv" 
INPUT_PATH = os.path.join(INPUT_DIR, TARGET_FILE)
MIN_COMMENTS_THRESHOLD = 10 # 忽略评论极少的周

os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_week_range(end_date):
    """
    辅助函数：将结束日期转换为 'YYYY-MM-DD ~ YYYY-MM-DD' 格式
    """
    start_date = end_date - pd.Timedelta(days=6)
    return f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

def weighted_aggregation(df_daily, freq):
    """
    通用加权聚合函数 (Daily -> Weekly/Monthly)
    """
    df = df_daily.copy()
    # 1. 还原绝对数量 (Mass)
    df['pos_count'] = df['pos_ratio'] * df['total_comments']
    df['neg_count'] = df['neg_ratio'] * df['total_comments']
    
    # 2. 重采样聚合
    resampled = df.resample(freq).sum(numeric_only=True)
    df_agg = resampled[resampled['total_comments'] > 0].copy()

    # 3. 重新计算加权平均比率
    df_agg['polarity_index'] = (df_agg['pos_count'] - df_agg['neg_count']) / df_agg['total_comments']
    df_agg['pos_ratio'] = df_agg['pos_count'] / df_agg['total_comments']
    df_agg['neg_ratio'] = df_agg['neg_count'] / df_agg['total_comments']

    # 4. 格式化日期/索引
    # 对于 CSV 输出，我们需要 Start~End 字符串
    # 但对于绘图，保留 Timestamp 对象最好（方便缩放），所以我们生成一个 string 列备用
    df_agg.reset_index(inplace=True)
    
    if freq == 'W-MON':
        df_agg['date_str'] = df_agg['date'].apply(format_week_range)
    else:
        df_agg['date_str'] = df_agg['date'].dt.strftime('%Y-%m')

    return df_agg.round(4)

def analyze_sentiment_peaks(df_weekly):
    """
    提取周度情感尖峰 (Top 20 最积极 & Top 20 最消极)
    """
    df_valid = df_weekly[df_weekly['total_comments'] >= MIN_COMMENTS_THRESHOLD].copy()
    
    # 1. 最积极
    pos_peaks = df_valid.nlargest(20, 'polarity_index').copy()
    pos_peaks['type'] = 'Positive Peak'
    
    # 2. 最消极
    neg_peaks = df_valid.nsmallest(20, 'polarity_index').copy()
    neg_peaks['type'] = 'Negative Peak'
    
    # 合并
    peaks = pd.concat([pos_peaks, neg_peaks])
    
    # 使用格式化好的日期范围列
    peaks = peaks.rename(columns={'date_str': 'week_range'})
    
    cols = ['week_range', 'type', 'polarity_index', 'total_comments', 'pos_ratio', 'neg_ratio']
    return peaks[cols].sort_values(['type', 'polarity_index'], ascending=False)

def plot_sentiment_history(df_monthly, df_weekly):
    """
    绘制综合情感图表
    上图: 月度情感极性 (红蓝柱状图)
    下图: 周度情感波动 (折线图)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("月度情感极性 (Monthly Polarity)", "周度情感波动 (Weekly Fluctuations)"),
        row_heights=[0.5, 0.5]
    )

    # === 上图: 月度趋势 (红蓝柱状图) ===
    # 逻辑: 正值用橙红色 (#E74C3C), 负值用蓝色 (#3498DB)
    colors = ['#E74C3C' if val >= 0 else '#3498DB' for val in df_monthly['polarity_index']]
    
    fig.add_trace(go.Bar(
        x=df_monthly['date'], 
        y=df_monthly['polarity_index'],
        name="月度极性",
        marker_color=colors,
        text=df_monthly['date_str'], # 鼠标悬停显示年月
        hoverinfo='x+y'
    ), row=1, col=1)

    # === 下图: 周度细节 (恢复折线图) ===
    fig.add_trace(go.Scatter(
        x=df_weekly['date'], 
        y=df_weekly['polarity_index'],
        name="周度趋势",
        mode='lines',
        line=dict(color='gray', width=1.5), # 使用灰色线条展示细节
        # 鼠标悬停显示 "YYYY-MM-DD ~ YYYY-MM-DD"
        customdata=df_weekly['date_str'],
        hovertemplate='<b>周度范围:</b> %{customdata}<br><b>极性:</b> %{y:.4f}<extra></extra>' 
    ), row=2, col=1)

    # 添加 0 轴参考线
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

    # 布局设置
    fig.update_layout(
        title="<b>Yamibo 论坛情感历史演变 (Sentiment History)</b>",
        template="plotly_white",
        hovermode="x unified",
        height=800,
        showlegend=False # 不需要图例，因为颜色和标题已经很清楚了
    )
    
    # 坐标轴
    fig.update_yaxes(title_text="情感指数", row=1, col=1)
    fig.update_yaxes(title_text="情感指数", row=2, col=1)

    output_path = os.path.join(OUTPUT_DIR, "sentiment_history_chart.html")
    fig.write_html(output_path)
    print(f"图表已生成: {output_path}")

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"错误: 找不到文件 {INPUT_PATH}")
        return

    print("Step 1: 读取并聚合数据...")
    df_daily = pd.read_csv(INPUT_PATH, parse_dates=['date'], index_col='date')

    # 生成周度/月度数据
    df_weekly = weighted_aggregation(df_daily, 'W-MON')
    df_monthly = weighted_aggregation(df_daily, 'ME')

    # 保存 CSV (将 date 列替换为 date_str 以满足需求)
    # 1. 保存月度
    csv_monthly = df_monthly.drop(columns=['date']).rename(columns={'date_str': 'month'})
    # 把 month 放到第一列
    cols = ['month'] + [c for c in csv_monthly.columns if c != 'month']
    csv_monthly[cols].to_csv(os.path.join(OUTPUT_DIR, "sentiment_monthly.csv"), index=False)
    
    # 2. 保存周度
    csv_weekly = df_weekly.drop(columns=['date']).rename(columns={'date_str': 'week_range'})
    cols = ['week_range'] + [c for c in csv_weekly.columns if c != 'week_range']
    csv_weekly[cols].to_csv(os.path.join(OUTPUT_DIR, "sentiment_weekly.csv"), index=False)
    
    print("已保存 CSV 数据 (日期格式已更新)。")

    print("Step 2: 分析情感尖峰...")
    peaks = analyze_sentiment_peaks(df_weekly)
    peaks_path = os.path.join(OUTPUT_DIR, "sentiment_peaks_list.csv")
    peaks.to_csv(peaks_path, index=False)
    print(f"情感尖峰列表已保存: {peaks_path}")

    print("Step 3: 绘制情感趋势图...")
    plot_sentiment_history(df_monthly, df_weekly)

if __name__ == "__main__":
    main()