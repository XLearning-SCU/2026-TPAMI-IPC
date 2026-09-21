import os
import torch
import torchvision
import numpy as np
from sklearn.cluster import KMeans

from utils import evaluation
from utils.evaluation import get_y_preds
from parse import get_parse
import clip
from utils.data import load_data
from utils import clevr4
import faiss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from our_pair_all import MyPair, RepeatedPair
from sklearn.metrics.pairwise import cosine_similarity
import random
from random_pair import sample_pairs



def tsne_pair(embedding, labels, name=None, pair=None, index=None):
    if isinstance(embedding, np.ndarray):
        new_features = embedding
    elif isinstance(embedding, torch.Tensor):
        new_features = embedding.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    if cluster_num <= 20:
        color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'lightblue',
                 'darkorange', 'lightgreen', 'firebrick', 'darkviolet', 'burlywood', 'lightcoral', 'darkgray', 'darkcyan',
                 'lightyellow']
        color = np.array(color)
    else:
        cmap = plt.get_cmap("tab20", 200)
        color = [cmap(i) for i in range(200)]

    img_labels = labels.astype(int)
    print(new_features.shape)
    x_new = TSNE(n_components=2).fit_transform(new_features)
    print(f"x_new.shape: {x_new.shape}")
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])

    for i in range(len(set(img_labels))):
        plt.scatter(x_new[img_labels == i, 0], x_new[img_labels == i, 1], color=color[i], marker='o', s=1, label=i)
    # plt.legend()


    if name is not None:
        plt.title(f"{name}")
    if pair is not None:
        for i, j, p_or_n in pair:
            if p_or_n:
                plt.plot([x_new[i, 0], x_new[j, 0]], [x_new[i, 1], x_new[j, 1]], c='black', linestyle='--', alpha=0.7)
            else:
                plt.plot([x_new[i, 0], x_new[j, 0]], [x_new[i, 1], x_new[j, 1]], c='black', alpha=0.7)

    if index is not None:
        plt.scatter(x_new[index, 0], x_new[index, 1], c=color[img_labels[index]], edgecolors='black', marker='o', s=100)

    # plt.savefig(f"{name}.png")
    plt.show()



def plot_tsne_with_anchor(features, anchor_idx):
    if isinstance(features, np.ndarray):
        features = features
    elif isinstance(features, torch.Tensor):
        features = features.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")
    tsne = TSNE(n_components=2)
    tsne_coords = tsne.fit_transform(features)

    # 计算和锚点的相似度（余弦相似度）
    anchor_feat = features[anchor_idx].reshape(1, -1)
    sims = cosine_similarity(features, anchor_feat).flatten()

    # 归一化到 [0,1]
    # sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    sims = (sims - sims.mean()) / sims.std()

    # 画图，颜色映射按相似度
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
        c=sims, cmap="viridis", s=1)
    # 标记锚点
    plt.scatter(tsne_coords[anchor_idx, 0], tsne_coords[anchor_idx, 1],
                c="red", s=100, marker="*", edgecolors="k", linewidths=1.5,
                label="Anchor")

    plt.colorbar(scatter, label="Similarity to Anchor")
    plt.legend()
    plt.title(f"{anchor_idx}")
    plt.show()





if __name__ == "__main__":
    args = get_parse()
    print(args)

    dataset_path, true_label, gpt_label, prompt = load_data(args)
    print(f"prompt: {prompt}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    save_path = os.path.join("./data", args.dataset, '')
    img_features = np.load(save_path + args.criterion + "_image_embedding.npy")
    img_features = torch.from_numpy(img_features).type(torch.float32)
    print(f"img_features.shape: {img_features.shape}")
    img_labels = np.loadtxt(save_path + args.criterion + "_labels.txt")
    print(f"img_labels.shape: {img_labels.shape}")

    cluster_num = len(set(img_labels))
    print(f"cluster_num: {cluster_num}")
    sample_num = len(img_labels)
    print(f"sample_num: {sample_num}")

    prediction_labels = np.loadtxt(save_path + args.criterion + "_prediction_vector.txt")
    print(f"prediction_labels.shape: {prediction_labels.shape}")

    idc_features = np.load(save_path + args.criterion + "_feature_idc.npy")
    idc_features = torch.from_numpy(idc_features).type(torch.float32)
    print(f"idc_features.shape: {idc_features.shape}")

    tsne_pair(idc_features, img_labels, None, None)
    exit(0)


    new_features = np.load(save_path + args.criterion + "_feature_vector.npy")
    new_features = torch.from_numpy(new_features).type(torch.float32)
    print(f"new_features.shape: {new_features.shape}")


    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    print(f"common_texts.shape: {common_texts.shape}")
    common_texts = torch.from_numpy(common_texts).type(torch.float32)

    sim_features = img_features @ common_texts.T
    print(f"sim_features.shape: {sim_features.shape}")
    sim_features /= torch.norm(sim_features, p=2, dim=1, keepdim=True)
    # sim_kmeans = KMeans(n_clusters=cluster_num, n_init=20, random_state=args.seed).fit(sim_features)
    # sim_labels = sim_kmeans.labels_

    pair_path = os.path.join(save_path, "our_pair", '')
    # complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-each" + ".tar"
    complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-high+all" + ".tar"
    
    print(f"complete_path: {complete_path}")
    selection = torch.load(complete_path, weights_only=False)
    selected_pair = selection['selected_pair']
    print("selected pairs num: ", len(selected_pair))
    num_p, num_n = selected_pair.get_p_or_n_len()
    print(f"num_p: {num_p}, num_n: {num_n}")


    # tsne_pair(sim_features, img_labels, complete_path, selected_pair)
    # for i in [14524, 6487, 23199, 44866, 49243, 16862, 58720, 22356]:
    #     plot_tsne_with_anchor(sim_features, anchor_idx=i)



    random_pairs = sample_pairs(sample_num, img_labels, args.pair_budget)
    tsne_pair(sim_features, img_labels, None, random_pairs)



    # tsne_pair(img_features, img_labels, None, None)

    # tsne_pair(sim_features, img_labels, None, selected_pair)

    # tsne_pair(new_features, img_labels, None, None)





































