import os
import numpy as np
import torch
import clip
from tqdm import tqdm
import torchvision
from parse import get_parse
from utils import evaluation
import random
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.data import load_data
import torch.nn as nn


def true_testing(embedding, true_text, true_label):
    print("zero-shot: using true_label")
    true_sim = (100 * embedding @ true_text.t()).softmax(dim=-1)
    true_prediction = torch.argmax(true_sim, dim=-1).numpy()
    nmi, ari, _, acc = evaluation.evaluate(true_label, true_prediction)
    print("nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))


def show_tsne(embedding, labels, index=None, name=None):
    if isinstance(embedding, np.ndarray):
        new_features = embedding
    elif isinstance(embedding, torch.Tensor):
        new_features = embedding.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    # color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'lightblue',
    #          'darkorange', 'lightgreen', 'firebrick', 'darkviolet', 'burlywood', 'lightcoral', 'darkgray', 'darkcyan',
    #          'lightyellow']
    # color = np.array(color)
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
        plt.scatter(x_new[img_labels == i, 0], x_new[img_labels == i, 1], color=color[i], marker='o', s=1,
                    label=i)
    # plt.legend()

    # plt.scatter(x_new[:, 0], x_new[:, 1], c='black', marker='o', s=1, alpha=1)
    if index is not None:
        plt.scatter(x_new[index, 0], x_new[index, 1], c='red', marker='o', s=100)
    if name is not None:
        plt.title(f"{name}")

    # plt.savefig(f"{name}.png")
    plt.show()


def kmeans_testing(embedding, k, true_label, process=True):
    if isinstance(embedding, np.ndarray):
        new_features = embedding
    elif isinstance(embedding, torch.Tensor):
        new_features = embedding.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    nmi_list, acc_list, ari_list = [], [], []
    for i in tqdm(range(20)):
        kmeans = KMeans(n_clusters=k, random_state=i, n_init='auto').fit(new_features)
        nmi, ari, _, acc = evaluation.evaluate(true_label, kmeans.labels_)
        if process:
            print("nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))
        nmi_list.append(nmi * 100)
        acc_list.append(acc * 100)
        ari_list.append(ari * 100)
    nmi_list = np.array(nmi_list)
    acc_list = np.array(acc_list)
    ari_list = np.array(ari_list)
    nmi_mean, acc_mean, ari_mean = np.mean(nmi_list), np.mean(acc_list), np.mean(ari_list)
    print(f"nmi_mean: {nmi_mean: .2f}, acc_mean: {acc_mean: .2f}, ari_mean: {ari_mean: .2f}")
    return nmi_mean, acc_mean, ari_mean




if __name__ == "__main__":
    args = get_parse()
    print(f"dataset: {args.dataset}")
    print(f"criterion: {args.criterion}")
    # print(args)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # clip_model, preprocess = clip.load('ViT-B/32', device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_path, true_label, gpt_label, prompt = load_data(args)
    print(f"prompt: {prompt}")
    # print(f"true_labels: {true_label}")
    # print(f"gpt_labels: {gpt_label}")



    if args.VLM == "ALIGN":
        save_path = os.path.join(BASE_DIR, "other_VLM/ALIGN", args.dataset, '')
    elif args.VLM == "MetaCLIP":
        save_path = os.path.join(BASE_DIR, "other_VLM/MetaCLIP", args.dataset, '')
    elif args.VLM == "BLIP2":
        save_path = os.path.join(BASE_DIR, "other_VLM/BLIP2", args.dataset, '')
    elif args.VLM == "CLIP":
        save_path = os.path.join(BASE_DIR, "data", args.dataset, '')
    else:
        raise NotImplementedError


    img_features = np.load(save_path + args.criterion + "_image_embedding.npy")
    img_features = torch.from_numpy(img_features).type(torch.float32)
    print(f"img_features.shape: {img_features.shape}")
    img_labels = np.loadtxt(save_path + args.criterion + "_labels.txt")
    print(f"img_labels.shape: {img_labels.shape}")
    print(img_labels)

    cluster_num = len(set(img_labels))
    print(f"cluster_num: {cluster_num}")
    sample_num = len(img_labels)
    print(f"sample_num: {sample_num}")

    kmeans_testing(img_features, cluster_num, img_labels, process=False)
    # exit(0)
    # show_tsne(img_features, img_labels)

    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    print(f"common_texts.shape: {common_texts.shape}")
    common_texts = torch.from_numpy(common_texts).type(torch.float32)

    # llama_texts = np.load(save_path + args.criterion + "_llama_texts.npy")
    # print(f"llama_texts.shape: {llama_texts.shape}")
    # llama_texts = torch.from_numpy(llama_texts).type(torch.float32)

    # qwen_texts = np.load(save_path + args.criterion + "_qwen_texts.npy")
    # print(f"qwen_texts.shape: {qwen_texts.shape}")
    # qwen_texts = torch.from_numpy(qwen_texts).type(torch.float32)


    sim_features = img_features @ common_texts.T

    # sim_features = img_features @ llama_texts.T

    # sim_features = img_features @ qwen_texts.T

    print(f"sim_features.shape: {sim_features.shape}")
    sim_features /= torch.norm(sim_features, p=2, dim=1, keepdim=True)
    print("sim_feature")
    kmeans_testing(sim_features, cluster_num, img_labels, process=False)
    # show_tsne(sim_features, img_labels)
    # exit(0)











