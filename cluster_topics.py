import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# === 尝试导入 adjustText 以解决文字重叠 ===
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("提示: 未检测到 adjustText 库，图片标签可能会重叠。建议运行 `pip install adjustText` 进行安装。")

# === 配置 ===
MODEL_PATH = "result/nlp/yamibo_word2vec.model"
RESULTS_DIR = "result/nlp/clusters"
FONT_PATH = "simhei.ttf" 

# 每次聚类维持的词总数
TOP_N_WORDS = 8000

# 聚类数
NUM_CLUSTERS = 250 

# 阈值：超过这个规模的类会被视为“噪音”
MAX_CLUSTER_SIZE = 160

# 最大迭代次数
MAX_ITERATIONS = 50

# 每迭代X轮画一次图
PLOT_FREQUENCY = 10

# 每一聚类在图中标注多少个词 (分布点数量)
WORD_DENSITY = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

def merge_class_vectors(model, words, labels):
    """
    类内大小写融合逻辑保持不变
    """
    print("  -> 正在执行类内大小写合并与向量平均...")
    
    clusters_data = {i: {} for i in range(NUM_CLUSTERS)}
    
    for word, label in zip(words, labels):
        lower_w = word.lower()
        if lower_w not in clusters_data[label]:
            clusters_data[label][lower_w] = {'originals': [], 'vectors': []}
        
        clusters_data[label][lower_w]['originals'].append(word)
        clusters_data[label][lower_w]['vectors'].append(model.wv[word])

    final_words = []
    final_labels = []
    final_vectors = []
    new_cluster_map = {i: [] for i in range(NUM_CLUSTERS)}

    for cid in range(NUM_CLUSTERS):
        for lower_w, data in clusters_data[cid].items():
            originals = data['originals']
            vecs = data['vectors']
            
            best_word = originals[0] 
            best_rank = float('inf')
            for w in originals:
                if w in model.wv.key_to_index:
                    rank = model.wv.key_to_index[w]
                    if rank < best_rank:
                        best_rank = rank
                        best_word = w
            
            if len(vecs) > 1:
                avg_vec = np.mean(vecs, axis=0)
                avg_vec = avg_vec / np.linalg.norm(avg_vec)
                final_vectors.append(avg_vec)
            else:
                final_vectors.append(vecs[0])
            
            final_words.append(best_word)
            final_labels.append(cid)
            new_cluster_map[cid].append(best_word)
            
    return np.array(final_words), np.array(final_labels), np.array(final_vectors), new_cluster_map

def visualize_clusters(vectors, words, labels, output_file):
    """
    核心功能优化: 均匀分布的标签选取
    """
    print("  -> 正在生成 t-SNE 可视化图表...")
    try:
        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        vectors_2d = tsne.fit_transform(vectors)
        
        plt.figure(figsize=(32, 32)) 
        font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None
        
        unique_labels = set(labels)
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
        
        texts_to_adjust = []

        for i, label in enumerate(unique_labels):
            # 获取该类的所有点
            idxs = [j for j, l in enumerate(labels) if l == label]
            if not idxs: continue
            
            coords = vectors_2d[idxs]
            # 绘制散点 (透明度稍微低点，突出文字)
            plt.scatter(coords[:, 0], coords[:, 1], c=[colors[i]], label=f'T{label}', alpha=0.4, s=40, edgecolors='none')
            
            # === 新增逻辑：基于子聚类的均匀采样 ===
            n_samples = len(idxs)
            # 如果点太少，就全部显示
            if n_samples <= WORD_DENSITY:
                target_indices = range(n_samples)
            else:
                # 使用 KMeans 将该类再次划分为 WORD_DENSITY 个子区域
                # 这样可以确保选出的点分布在簇的各个部分，而不是挤在中心
                try:
                    sub_kmeans = KMeans(n_clusters=WORD_DENSITY, random_state=42, n_init=5)
                    sub_labels = sub_kmeans.fit_predict(coords)
                    
                    target_indices = []
                    # 对于每个子区域，找离该子区域中心最近的点
                    for sub_id in range(WORD_DENSITY):
                        sub_idxs = np.where(sub_labels == sub_id)[0]
                        if len(sub_idxs) == 0: continue
                        
                        sub_center = sub_kmeans.cluster_centers_[sub_id]
                        sub_coords = coords[sub_idxs]
                        
                        # 计算距离
                        dists = np.sum((sub_coords - sub_center)**2, axis=1)
                        nearest_sub_idx = np.argmin(dists)
                        
                        # 转换回局部索引
                        target_indices.append(sub_idxs[nearest_sub_idx])
                        
                except Exception as e:
                    # 如果子聚类失败（极少见），回退到简单中心采样
                    center = np.mean(coords, axis=0)
                    dists = np.sum((coords - center)**2, axis=1)
                    target_indices = np.argsort(dists)[:WORD_DENSITY]
            
            # 绘制选定的标签
            for local_idx in target_indices:
                real_idx = idxs[local_idx]
                word = words[real_idx]
                x, y = vectors_2d[real_idx, 0], vectors_2d[real_idx, 1]
                
                txt = plt.text(x, y, word, 
                               fontproperties=font, 
                               fontsize=10, 
                               fontweight='bold', 
                               color='black',
                               alpha=0.9)
                texts_to_adjust.append(txt)

        plt.title(f"Topic Clusters (Distributed Labels)", fontsize=24)
        
        if HAS_ADJUST_TEXT and texts_to_adjust:
            print(f"  -> 正在优化 {len(texts_to_adjust)} 个标签的位置... (请耐心等待)")
            adjust_text(texts_to_adjust, 
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=5, shrinkB=5),
                        force_text=(0.9, 0.9), 
                        force_points=(0.3, 0.6) 
                        )
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> 可视化保存成功: {output_file}")
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()

def get_valid_words(model, top_n, banned_set):
    """从模型中提取 Top N 个未被封禁的词"""
    valid_words = []
    # index_to_key 已经按词频降序排列
    for word in model.wv.index_to_key:
        if word not in banned_set:
            valid_words.append(word)
            if len(valid_words) >= top_n:
                break
    return valid_words

def save_results(cluster_map, iteration):
    """保存当前轮次的结果"""
    output_csv = os.path.join(RESULTS_DIR, "topic_clusters.csv")
    output_txt = os.path.join(RESULTS_DIR, "topic_clusters_readable.txt")
    
    rows = []
    total_words = sum(len(v) for v in cluster_map.values())
    
    with open(output_txt, "w", encoding="utf-8") as f_txt:
        f_txt.write(f"=== Final Clusters (Result of Iteration {iteration}) ===\n")
        f_txt.write(f"Total Unique Words: {total_words}\n\n")
        
        for cluster_id in sorted(cluster_map.keys()):
            words_in_cluster = cluster_map[cluster_id]
            f_txt.write(f"=== Topic {cluster_id:02d} ({len(words_in_cluster)} words) ===\n")
            f_txt.write(" ".join(words_in_cluster) + "\n\n")
            
            rows.append({
                "cluster_id": cluster_id,
                "keywords": ", ".join(words_in_cluster),
                "count": len(words_in_cluster)
            })
    
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"  -> 结果已更新保存至: {RESULTS_DIR}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    print("正在加载 Word2Vec 模型...")
    model = Word2Vec.load(MODEL_PATH)
    
    banned_words = set()
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n========== 第 {iteration} 轮迭代 ==========")
        
        # 1. 补充词库 (Refill)
        current_words = get_valid_words(model, TOP_N_WORDS, banned_words)
        if len(current_words) < NUM_CLUSTERS:
            print("可用词不足，无法继续聚类。停止。")
            break
            
        print(f"当前聚类词数: {len(current_words)}")
        
        # 2. 聚类
        raw_vectors = model.wv[current_words]
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(raw_vectors)
        
        # 3. 类内大小写融合
        merged_words, merged_labels, merged_vectors, cluster_map = merge_class_vectors(model, current_words, labels)
        
        # 4. 识别需要剔除的类
        clusters_to_ban = []
        overflow_clusters = [cid for cid, words in cluster_map.items() if len(words) > MAX_CLUSTER_SIZE]
        
        if overflow_clusters:
            print(f"  策略 A: 发现 {len(overflow_clusters)} 个类规模超过阈值 ({MAX_CLUSTER_SIZE})。")
            clusters_to_ban = overflow_clusters
        else:
            print(f"  策略 B: 所有类均在阈值内。强制剔除当前规模最大的 1 个类以继续收敛...")
            largest_cid = max(cluster_map, key=lambda k: len(cluster_map[k]))
            clusters_to_ban = [largest_cid]

        # 5. 执行剔除
        print("  -> 正在执行剔除操作:")
        count_removed = 0
        clusters_to_ban.sort(key=lambda x: len(cluster_map[x]), reverse=True)
        
        for cid in clusters_to_ban:
            words_to_remove = cluster_map[cid] 
            count_removed += len(words_to_remove)
            
            sample_len = 10
            sample_words = words_to_remove[:sample_len]
            sample_str = ", ".join(sample_words)
            ellipsis = "..." if len(words_to_remove) > sample_len else ""
            
            print(f"     [X] 剔除 Topic {cid:03d} (规模: {len(words_to_remove)}): {sample_str} {ellipsis}")
            
            banned_words.update(words_to_remove)
            
        print(f"  -> 本轮共剔除 {count_removed} 个词。")
        
        # 6. 保存每一轮结果
        save_results(cluster_map, iteration)
        
        # 7. 可视化
        if iteration % PLOT_FREQUENCY == 0 or iteration == MAX_ITERATIONS:
            visualize_clusters(merged_vectors, merged_words, merged_labels, os.path.join(RESULTS_DIR, f"viz_iter_{iteration}.png"))

    print(f"\n达到最大迭代次数 ({MAX_ITERATIONS})，流程结束。")

if __name__ == "__main__":
    main()