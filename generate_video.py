import bar_chart_race as bcr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import warnings

# 字体配置
font_path = 'simhei.ttf'

# 1. 屏蔽警告
warnings.filterwarnings("ignore")

# 2. 强制加载字体
if not os.path.exists(font_path):
    print(f"错误：找不到 {font_path}，请确保文件在当前目录下。")
    exit()
    
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

print("正在读取数据...")
# 3. 读取数据
df = pd.read_csv("result/viz/cumulative_general_all.csv", index_col='date', parse_dates=['date'])

# === 核心优化：数据重采样 (控制时长的关键) ===
# 原始数据是每天一行，20年有7000多帧，导致视频极长。
# 这里改为 '3D' (每3天一帧) 或 'W' (每周一帧)
# '3D' 大约能把时长压缩到原来的 1/3 (约7分钟)
# '5D' 大约能压缩到 1/5 (约4分钟)
print("正在重采样数据以缩短视频时长...")
df_resampled = df.resample('W').last().ffill() 

print(f"原始帧数: {len(df)}, 重采样后帧数: {len(df_resampled)}")

# 4. 生成视频
print("开始生成视频 (这可能需要几分钟)...")
bcr.bar_chart_race(
    df=df_resampled,
    filename='yamibo_history.mp4',
    
    # === 动画参数 ===
    n_bars=15,             # 显示前15名
    period_length=120,     # 每帧停留毫秒数 (越小越快)
    steps_per_period=10,   # 过渡帧数 (越小动画越生硬但生成越快，10比较丝滑)
    
    # === 样式美化 ===
    cmap='tab20',          # 配色方案，可选: 'dark12', 'prism', 'Set1', 'tab20'
    bar_label_size=7,      # 柱子上数字的大小
    tick_label_size=7,     # y轴文字大小
    
    # === 字体配置 ===
    shared_fontdict={
        'family': my_font.get_name(), 
        'weight': 'bold',
        'color': '#333333', 
    },
    
    # === 标题与标签 ===
    title='百合会论坛历史热词 (2004-2025)',
    bar_size=.95,          # 柱子宽度
    
    # 优化日期显示位置和格式
    period_label={
        'x': .95, 'y': .15, 
        'ha': 'right', 'va': 'center',
        'size': 16,
        'weight': 'bold',
        'color': '#555555'
    },
    period_fmt='%Y-%m-%d', # 格式化日期
    
    # === 性能参数 ===
    dpi=144,               # 清晰度
    filter_column_colors=True # 减少颜色重复警告
)

print(f"视频生成完成！请查看 'yamibo_history.mp4'")