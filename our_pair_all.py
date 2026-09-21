import os
import torch
import torchvision
import numpy as np
from utils import evaluation
from utils.evaluation import get_y_preds
from parse import get_parse
import clip
from utils.data import load_data
from utils.other_llm import data_ablation
from utils import clevr4
import faiss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt




class MyPair:
    def __init__(self):
        self._pairs = set()

    def add(self, i, j, p_or_n):
        """p_or_n表示这一对是同一类还是不同类"""
        if i == j:
            raise ValueError("Pair elements must be distinct")
        self._pairs.add((min(i, j), max(i, j), p_or_n))

    def get_related_indices(self, idx):
        """获取所有与 idx 配对的索引"""
        related = []
        for a, b, _ in self._pairs:
            if a == idx:
                related.append(b)
            elif b == idx:
                related.append(a)
        return related

    def get_p_or_n_len(self):
        num_p = 0
        num_n = 0
        for i, j, p_or_n in self._pairs:
            if p_or_n:
                num_p += 1
            else:
                num_n += 1
        return num_p, num_n


    def __contains__(self, pair):
        """Allow 'in' operator usage: (i,j) in selected_pair"""
        i, j, p_or_n = pair
        return (min(i, j), max(i, j), p_or_n) in self._pairs

    def __len__(self):
        """Return the number of pairs stored"""
        return len(self._pairs)

    def __iter__(self):
        """Allow iteration over the pairs"""
        return iter(self._pairs)

    def __str__(self):
        """String representation of the collection"""
        return str(self._pairs)


class RepeatedPair:
    def __init__(self):
        self._pairs = set()

    def add(self, i, j, p_or_n):
        """p_or_n表示这一对是同一类还是不同类"""
        if i == j:
            raise ValueError("Pair elements must be distinct")
        self._pairs.add((i, j, p_or_n))

    def get_related_indices(self, idx):
        """获取所有与 idx 配对的索引"""
        related = []
        for a, b, _ in self._pairs:
            if a == idx:
                related.append(b)
            elif b == idx:
                related.append(a)
        return related

    def get_p_or_n_len(self):
        num_p = 0
        num_n = 0
        for i, j, p_or_n in self._pairs:
            if p_or_n:
                num_p += 1
            else:
                num_n += 1
        return num_p, num_n


    def __contains__(self, pair):
        """Allow 'in' operator usage: (i,j) in selected_pair"""
        i, j, p_or_n = pair
        return (i, j, p_or_n) in self._pairs

    def __len__(self):
        """Return the number of pairs stored"""
        return len(self._pairs)

    def __iter__(self):
        """Allow iteration over the pairs"""
        return iter(self._pairs)

    def __str__(self):
        """String representation of the collection"""
        return str(self._pairs)


def kNN(x_train, x_test, K):
    assert len(x_train.shape) == 2
    assert len(x_test.shape) == 2
    N1, N2 = x_test.size(0), x_train.size(0)
    x_1 = torch.pow(x_test, 2).sum(1, keepdim=True).expand(N1, N2)
    x_2 = torch.pow(x_train, 2).sum(1, keepdim=True).expand(N2, N1).t()
    dist = x_1 + x_2
    dist.addmm_(x_test, x_train.t(), beta=1, alpha=-2)
    d_knn, ind_knn = torch.topk(dist, k=K, dim=1, largest=False, sorted=False)
    return ind_knn, d_knn


def partitioned_kNN(feats_list, K=20, partitions_size=130000):
    partitions = int(np.ceil(feats_list.shape[0] / partitions_size))
    # print("Partitions:", partitions)

    # Assume the last partition has at least K elements
    ind_knns = torch.zeros(
        (feats_list.size(0), partitions * K), dtype=torch.long)
    d_knns = torch.zeros(
        (feats_list.size(0), partitions * K), dtype=torch.float)

    def get_sampled_data(ind):
        return feats_list[ind * partitions_size: (ind + 1) * partitions_size]

    for ind_i in range(partitions):  # ind_i: train dimension
        for ind_j in range(partitions):  # ind_j: test dimension
            # print("Running with indices: {}, {}".format(ind_i, ind_j))
            x_train = get_sampled_data(ind_i).cuda()
            x_test = get_sampled_data(ind_j).cuda()

            ind_knn, d_knn = kNN(x_train, x_test, K=K)
            # ind_knn, d_knn: test dimension, K (indices: train dimension)
            ind_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size, ind_i * K: (ind_i + 1) * K] = \
                ind_i * partitions_size + ind_knn.cpu()
            d_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size,
            ind_i * K: (ind_i + 1) * K] = d_knn.cpu()

            del ind_knn, d_knn, x_train, x_test

    d_sorted_inds = d_knns.argsort(dim=1)
    d_selected_inds = d_sorted_inds[:, :K]
    ind_knns_selected = torch.gather(
        ind_knns, dim=1, index=d_selected_inds)
    d_knns_selected = torch.gather(d_knns, dim=1, index=d_selected_inds)
    d_knns = d_knns_selected
    ind_knns = ind_knns_selected

    del ind_knns_selected, d_knns_selected

    return d_knns, ind_knns


def point_selection(matrix, hardness, budget, alpha, beta, gamma):
    selected_index = []
    remaining_index = torch.arange(matrix.shape[0])
    d_knns, ind_knns = partitioned_kNN(matrix)
    neighbors_dist = d_knns.mean(dim=1)
    representativeness = torch.log(1 / neighbors_dist)
    origin_score = representativeness * alpha + beta * hardness
    for i in range(budget):
        if i == 0:
            score = origin_score
        else:
            remaining_matrix = matrix[remaining_index]
            selected_matrix = matrix[selected_index]
            diversity = -torch.matmul(remaining_matrix, selected_matrix.t())
            diversity = torch.log(1+diversity)
            diversity = diversity.view(diversity.shape[0], -1)
            diversity = torch.min(diversity, dim=1).values
            score = origin_score[remaining_index] + gamma * diversity
        temp_index = torch.argmax(score)
        current_index = remaining_index[temp_index].item()
        # print("---------第%d个：%d" % (i + 1, current_index))
        # print("origin_score: %.2f, rep: %.2f, hard: %.2f" % (origin_score[current_index], representativeness[current_index], beta*hardness[current_index]))
        # if i != 0:
        #     print("diversity: ", gamma * diversity[temp_index])
        selected_index.append(current_index)
        # print("selected_index: ", selected_index)
        remaining_index = torch.cat((remaining_index[:temp_index], remaining_index[temp_index + 1:]))
        # print("remaining_index.shape: ", remaining_index.shape)
    return selected_index




def tsne_pair(embedding, labels, index=None, name=None, pair=None):
    if isinstance(embedding, np.ndarray):
        new_features = embedding
    elif isinstance(embedding, torch.Tensor):
        new_features = embedding.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    if class_num < 20:
        color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'lightblue',
                 'darkorange', 'lightgreen', 'firebrick', 'darkviolet', 'burlywood', 'lightcoral', 'darkgray', 'darkcyan',
                 'lightyellow']
        color = np.array(color)
    else:
        cmap = plt.get_cmap("tab20", 200)
        color = [cmap(i) for i in range(200)]
        color = np.array(color)

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

    if index is not None:
        plt.scatter(x_new[index, 0], x_new[index, 1], color=color[img_labels[index]], edgecolors='black', marker='o', s=100)

    if name is not None:
        plt.title(f"{name}")
    if pair is not None:
        for i, j, p_or_n in pair:
            if p_or_n:
                plt.plot([x_new[i, 0], x_new[j, 0]], [x_new[i, 1], x_new[j, 1]], color='red')
            else:
                plt.plot([x_new[i, 0], x_new[j, 0]], [x_new[i, 1], x_new[j, 1]], color='black')

    # plt.savefig(f"{name}.png")
    plt.show()




if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    args = get_parse()
    print(args)
    dataset_path, true_label, gpt_label, prompt = load_data(args)
    # dataset_path, true_label, gpt_label, prompt = data_ablation(args)


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
    # print(f"img_features.shape: {img_features.shape}")
    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    # llama_texts = np.load(save_path + args.criterion + "_llama_texts.npy")
    # qwen_texts = np.load(save_path + args.criterion + "_qwen_texts.npy")
    img_labels = np.loadtxt(save_path + args.criterion + "_labels.txt")
    img_labels = img_labels.astype(int)
    # print(f"img_labels.shape: {img_labels.shape}")

    class_num = len(set(img_labels))
    print(f"class_num: {class_num}")
    sample_num = len(img_labels)
    print(f"sample_num: {sample_num}")

    common_texts = torch.from_numpy(common_texts).type(torch.float32)
    # llama_texts = torch.from_numpy(llama_texts).type(torch.float32)
    # qwen_texts = torch.from_numpy(qwen_texts).type(torch.float32)
    img_features = torch.from_numpy(img_features).type(torch.float32)

    new_features = img_features @ common_texts.T
    # new_features = img_features @ llama_texts.T
    # new_features = img_features @ qwen_texts.T
    print(f"new_features.shape: {new_features.shape}")

    new_features /= torch.norm(new_features, p=2, dim=1, keepdim=True)
    # new_features = new_features - torch.mean(new_features)
    # new_features /= torch.std(new_features, dim=1, keepdim=True)
    new_features = new_features.numpy()

    print("------------Performing faiss k-means clustering------------")
    faiss_kmeans = faiss.Kmeans(d=new_features.shape[1], k=class_num, niter=100, verbose=False, gpu=False, nredo=20, seed=args.seed)
    faiss_kmeans.train(new_features)

    centroids = faiss_kmeans.centroids
    distances_1, kmeans_labels = faiss_kmeans.index.search(new_features, 1)  # 每个样本的最近中心
    kmeans_labels = kmeans_labels.reshape(-1)
    nmi, ari, _, acc = evaluation.evaluate(img_labels, kmeans_labels)
    print("new nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))

    cos_sim = np.zeros((sample_num, class_num))
    for i in range(class_num):
        cos_sim[:, i] = np.dot(new_features, centroids[i]) / np.linalg.norm(centroids[i])
    # print(f"distances.shape: {cos_sim.shape}")

    nearing_top = np.argsort(-cos_sim, axis=1)

    hard_each = np.zeros(sample_num)
    for i in range(sample_num):
        hard_each[i] = cos_sim[i, nearing_top[i, 0]] - cos_sim[i, nearing_top[i, 1]]

    hard_each = 1 - hard_each
    hard_each = torch.from_numpy(hard_each)
    hard_each = torch.log(hard_each)
    hard_each = (hard_each - torch.min(hard_each)) / (torch.max(hard_each) - torch.min(hard_each))

    # 选出高价值样本点
    new_features = torch.from_numpy(new_features)

    if args.dataset == "fruit360" or args.dataset == "cifar10":
        args.pair_budget = 50
    else:
        args.pair_budget = 500

    each_num = args.each_num
    high_value_num = args.pair_budget // each_num
    print(f"挑选高价值点数量：{high_value_num}")

    selection_sample = point_selection(new_features, hard_each, budget=high_value_num, alpha=1., beta=0., gamma=1.)
    selected_sample = torch.tensor(selection_sample)



    # 选出高价值样本对
    all_cos = new_features @ new_features.T

    pair_cos = (all_cos - torch.mean(all_cos)) / torch.std(all_cos)




    # 全局算一个阈值
    # sorted_values, _ = torch.sort(pair_cos[selected_sample].reshape(-1), descending=True)
    sorted_values, _ = torch.sort(pair_cos.reshape(-1), descending=True)
    threshold = 1/class_num
    # threshold = 0.1
    index_percent = int(threshold * len(sorted_values))
    value_percent = sorted_values[index_percent]

    pair_threshold = value_percent
    print(f"pair_threshold: {pair_threshold}")

    pair_hard = 100 - torch.abs(pair_cos - pair_threshold)






    # for seed in range(5):
        # selection_sample = torch.randperm(sample_num)[:high_value_num]
        # selected_sample = selection_sample



    mask1 = torch.zeros_like(all_cos)
    mask1[selected_sample, :] = 1
    pair_all_hard = pair_hard * mask1

    for i in selected_sample:
        pair_all_hard[i, i] = 0

    selected_pair = RepeatedPair()
    remain_num = args.pair_budget - each_num * high_value_num

    for i in range(remain_num):
        idx_1 = selection_sample[i]

        # candidates = torch.arange(sample_num)
        # candidates = candidates[candidates != idx_1]
        # topK_indices = candidates[torch.randperm(len(candidates))[:each_num + 1]]

        _, topK_indices = torch.topk(pair_all_hard[idx_1].view(-1), each_num + 1)
        for idx_k in topK_indices:
            if img_labels[idx_1] == img_labels[idx_k.item()]:
                selected_pair.add(idx_k.item(), idx_1, 1)
            else:
                selected_pair.add(idx_k.item(), idx_1, 0)

    for i in range(high_value_num - remain_num):
        temp_i = i + remain_num
        idx_1 = selection_sample[temp_i]

        # candidates = torch.arange(sample_num)
        # candidates = candidates[candidates != idx_1]
        # topK_indices = candidates[torch.randperm(len(candidates))[:each_num]]

        _, topK_indices = torch.topk(pair_all_hard[idx_1].view(-1), each_num)
        for idx_k in topK_indices:
            if img_labels[idx_1] == img_labels[idx_k.item()]:
                selected_pair.add(idx_k.item(), idx_1, 1)
            else:
                selected_pair.add(idx_k.item(), idx_1, 0)



    # print(selected_pair)
    num_p, num_n = selected_pair.get_p_or_n_len()
    print(f"num_p: {num_p}, num_n: {num_n}")


    # tsne_pair(new_features, img_labels, selection_sample, args.criterion, selected_pair)
    # exit(0)


    pair_path = os.path.join(save_path, "our_pair", '')
    if not os.path.isdir(pair_path):
        os.makedirs(pair_path)

    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-llama" + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-qwen" + ".tar"
    save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-high+all" + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-img" + ".tar"

    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-rep+hard" + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-div+hard" + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-hard-" + str(seed) + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-rep-" + str(seed) + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-div-" + str(seed) + ".tar"
    # save_name = args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-rep+div-" + str(seed) + ".tar"



    our_selection = {'selected_pair': selected_pair}
    torch.save(our_selection, pair_path + save_name)
    print(f"saved to {pair_path + save_name}")






























