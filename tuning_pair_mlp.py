import os
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from parse import get_parse
from utils import evaluation
import random
from sklearn.cluster import KMeans
from utils.data import load_data
from utils.other_llm import data_ablation
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.model import MLP
import faiss
import math
import datetime
from our_pair_all import RepeatedPair, MyPair


class SelectedPairDataset(Dataset):
    def __init__(self, features, selected_pair):
        self.features = features
        self.pairs = list(selected_pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, p_or_n = self.pairs[idx]
        sample_i = self.features[i]
        sample_j = self.features[j]
        return sample_i, sample_j, p_or_n


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx



# 大于threshold_in选入，小于threshold_out淘汰
def generate_easy_samples(c, pseudo_label_cur, threshold_out, threshold_in, class_num):
    gamma = 0.2

    batch_size = c.shape[0]
    pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
    tmp = torch.arange(0, batch_size).to(device)

    prediction = c.argmax(dim=1)
    confidence = c.max(dim=1).values
    unconfident_pred_index = confidence < threshold_out
    pseudo_per_class = np.ceil(batch_size / class_num * gamma).astype(int)

    for i in range(class_num):
        class_idx = prediction == i
        if class_idx.sum() == 0:
            continue
        confidence_class = confidence[class_idx]
        num = min(confidence_class.shape[0], pseudo_per_class)
        confident_idx = torch.argsort(-confidence_class)
        for j in range(num):
            idx = tmp[class_idx][confident_idx[j]]
            if confidence[idx] > threshold_in:
                pseudo_label_nxt[idx] = i
            else:
                break

    todo_index = pseudo_label_cur == -1
    pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
    pseudo_label_nxt = pseudo_label_cur
    pseudo_label_nxt[unconfident_pred_index] = -1
    return pseudo_label_nxt



def symmetric_kl(p, q, eps=1e-7):
    kl_pq = torch.sum(p * torch.log((p + eps) / (q + eps)), dim=1)
    kl_qp = torch.sum(q * torch.log((q + eps) / (p + eps)), dim=1)
    return 0.5 * (kl_pq + kl_qp)



def pair_tuning_high(model, device, text=None):
    selected_epoch = 0
    high_epoch = 0
    loss_epoch = 0

    # for img_i, img_j, p_or_n in tqdm(selected_loader):
    for img_i, img_j, p_or_n in selected_loader:
        img_i, img_j, p_or_n = img_i.to(device), img_j.to(device), p_or_n.to(device)
        logits_i = model(img_i)
        logits_j = model(img_j)

        prob_i = F.softmax(logits_i, dim=1)
        prob_j = F.softmax(logits_j, dim=1)

        # 为避免数值稳定性问题，加上eps
        eps = 1e-7
        # 对称交叉熵
        ce_positive = -0.5 * (torch.sum(prob_i * torch.log(prob_j + eps), dim=1) +
                              torch.sum(prob_j * torch.log(prob_i + eps), dim=1))

        ce_negative = -0.5 * (torch.sum(prob_i * torch.log(1 - prob_j + eps), dim=1) +
                              torch.sum(prob_j * torch.log(1 - prob_i + eps), dim=1))

        p_or_n = p_or_n.view(-1).float()

        # 最终 loss: 同类用 ce_positive，不同类用 ce_negative
        selected_loss = p_or_n * ce_positive + (1 - p_or_n) * ce_negative

        # selected_loss = p_or_n * ce_positive

        selected_loss = selected_loss.mean()
        selected_epoch += selected_loss.item()

        unlabeled_iter = iter(all_loader)
        try:
            img, _, unlabeled_index = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(all_loader)
            img, _, unlabeled_index = next(unlabeled_iter)

        img = img.to(device)

        model.eval()
        with torch.no_grad():
            # 动态更新高置信度样本，初始化的时候用kmeans的预测跑几轮
            logits = model(img)
            prob = F.softmax(logits, dim=1)
            pseudo_label_cur = generate_easy_samples(prob, pseudo_label[unlabeled_index], threshold_in=args.high_threshold, threshold_out=args.high_threshold, class_num=cluster_num)
            pseudo_label[unlabeled_index] = pseudo_label_cur
            index_cur = pseudo_label_cur != -1

        model.train()

        # loss = selected_loss


        if index_cur.sum().item() < 2:
            loss = selected_loss
        else:
            logits_cur = model(img[index_cur])
            label_cur = pseudo_label_cur[index_cur].to(device).to(torch.long)
            idx, counts = torch.unique(label_cur, return_counts=True)
            freq = label_cur.shape[0] / counts.float()
            weight = torch.ones(cluster_num).to(device)
            weight[idx] = freq
            high_loss = torch.nn.functional.cross_entropy(logits_cur, label_cur, weight=weight)
            high_epoch += high_loss.item()

            loss = selected_loss + high_loss

        loss_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    high_confidence_index = pseudo_label != -1
    high_confidence_index = high_confidence_index.cpu()
    high_confidence_label = pseudo_label[high_confidence_index].cpu()

    idx_first, counts_first = torch.unique(high_confidence_label, return_counts=True)
    increase_first = torch.zeros(cluster_num)
    increase_first[idx_first] = counts_first.cpu().float()
    # print(f"high_confidence: {increase_first}")
    # print(f"high_confidence_num: {high_confidence_label.shape[0]}")
    # print(f"Selected Loss: {selected_epoch}, High Loss: {high_epoch}")



def high_tuning(model, device, epoch, text=None):
    high_epoch = 0
    # for img, _, unlabeled_index in tqdm(all_loader):
    for img, _, unlabeled_index in all_loader:
        img = img.to(device)

        temp = -torch.ones(img.shape[0], dtype=torch.long).to(device)
        index_cur = temp == -1
        pseudo_label_cur = pseudo_label[unlabeled_index]

        model.train()

        logits_cur = model(img[index_cur])
        label_cur = pseudo_label_cur[index_cur].to(device).to(torch.long)
        idx, counts = torch.unique(label_cur, return_counts=True)
        freq = label_cur.shape[0] / counts.float()
        weight = torch.ones(cluster_num).to(device)
        weight[idx] = freq
        high_loss = torch.nn.functional.cross_entropy(logits_cur, label_cur, weight=weight)
        high_epoch += high_loss.item()

        optimizer.zero_grad()
        high_loss.backward()
        optimizer.step()
    # print(f"High Loss: {high_epoch}")





def kmeans_testing(embedding, k, true_label, process=True):
    if isinstance(embedding, np.ndarray):
        new_features = embedding
    elif isinstance(embedding, torch.Tensor):
        new_features = embedding.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    nmi_list, acc_list, ari_list = [], [], []
    for i in range(20):
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
    # print(args)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_path, true_label, gpt_label, prompt = load_data(args)
    # dataset_path, true_label, gpt_label, prompt = data_ablation(args)
    print(f"prompt: {prompt}")



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

    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    print(f"common_texts.shape: {common_texts.shape}")
    common_texts = torch.from_numpy(common_texts).type(torch.float32)
    sim_features = img_features @ common_texts.T

    # llama_texts = np.load(save_path + args.criterion + "_llama_texts.npy")
    # print(f"llama_texts.shape: {llama_texts.shape}")
    # llama_texts = torch.from_numpy(llama_texts).type(torch.float32)
    # sim_features = img_features @ llama_texts.T    

    # qwen_texts = np.load(save_path + args.criterion + "_qwen_texts.npy")
    # print(f"qwen_texts.shape: {qwen_texts.shape}")
    # qwen_texts = torch.from_numpy(qwen_texts).type(torch.float32)
    # sim_features = img_features @ qwen_texts.T   


    print(f"sim_features.shape: {sim_features.shape}")
    # train_features = sim_features - torch.mean(sim_features)
    # train_features /= torch.std(train_features, dim=1, keepdim=True)

    cluster_features = sim_features / torch.norm(sim_features, p=2, dim=1, keepdim=True)

    # train_features = img_features
    train_features = cluster_features


    if args.dataset == "stanford-cars" or args.dataset == "cub":
        args.warm_up = 20
    else:
        args.warm_up = 10
    if args.dataset == "fruit360" or args.dataset == "cifar10":
        args.pair_budget = 50
    else:
        args.pair_budget = 500


    if args.random == 1:
        pair_path = os.path.join(save_path, "random_pair", '')
        complete_path = pair_path + str(args.pair_budget) + "-" + str(args.criterion) + "-" + str(args.seed) + ".tar"
    elif args.random == 0:
        pair_path = os.path.join(save_path, "our_pair", '')

        if args.strategy == 'rep+div+hard':
            # complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-llama"+ ".tar"
            # complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-qwen"+ ".tar"
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-high+all"+ ".tar"
        if args.strategy == 'rep+hard':
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-rep+hard" + ".tar"
        if args.strategy == 'div+hard':
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-div+hard" + ".tar"
        if args.strategy == 'hard':
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-hard-" + str(args.seed) + ".tar"
        if args.strategy == 'rep':
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-rep-" + str(args.seed) + ".tar"
        if args.strategy == 'div':
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-div-" + str(args.seed) + ".tar"
        if args.strategy == 'rep+div':
            complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-" + str(args.each_num) + "-rep+div-" + str(args.seed) + ".tar"


    elif args.random == -1:
        pair_path = os.path.join(save_path, "hard+div", '')
        # complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + ".tar"
        complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + "-label" + str(args.ratio) + "-change" + str(args.change_cluster) + ".tar"
    elif args.random == -2:
        pair_path = os.path.join(save_path, "hard", '')
        complete_path = pair_path + args.criterion + "-" + str(args.pair_budget) + ".tar"
    else:
        raise NotImplementedError
    print(f"complete_path: {complete_path}")
    selection = torch.load(complete_path, weights_only=False)
    selected_pair = selection['selected_pair']
    print("selected pairs num: ", len(selected_pair))
    num_p, num_n = selected_pair.get_p_or_n_len()
    print(f"num_p: {num_p}, num_n: {num_n}")

    # 给 selected_pair 加噪：按 noise_ratio 比例翻转 p_or_n
    if hasattr(args, 'noise_ratio') and args.noise_ratio > 0:
        print(f"noise_ratio: {args.noise_ratio*100}%")
        pairs_list = list(selected_pair._pairs)
        num_noise = int(len(pairs_list) * args.noise_ratio)
        noise_indices = set(random.sample(range(len(pairs_list)), num_noise))
        new_pairs = set()
        for idx, (i, j, p_or_n) in enumerate(pairs_list):
            if idx in noise_indices:
                new_pairs.add((i, j, 1 - p_or_n))
            else:
                new_pairs.add((i, j, p_or_n))
        selected_pair._pairs = new_pairs
        # num_p, num_n = selected_pair.get_p_or_n_len()
        # print(f"After noise (ratio={args.noise_ratio}): num_p: {num_p}, num_n: {num_n}")


    batch_size = min(64, args.pair_budget)
    dataset = FeatureDataset(train_features, img_labels)
    all_loader = DataLoader(dataset, batch_size=5*batch_size, shuffle=True, drop_last=False, num_workers=4)
    selected_dataset = SelectedPairDataset(train_features, selected_pair)
    selected_loader = DataLoader(selected_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

    test_dataloader = DataLoader(
        dataset,
        batch_size=5 * batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    model = MLP(train_features.shape[1], cluster_num)
    model.to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.epochs

    # 用k-means结果初始化（一开始所有的都当作是高置信度的）
    print("--------------------------------------")
    print("Performing faiss k-means clustering...")
    faiss_kmeans = faiss.Kmeans(d=train_features.shape[1], k=cluster_num, niter=100, verbose=False, gpu=False, nredo=20, seed=args.seed)
    faiss_kmeans.train(train_features.numpy())
    centroids = faiss_kmeans.centroids
    distances_1, kmeans_labels = faiss_kmeans.index.search(train_features.numpy(), 1)  # 每个样本的最近中心
    kmeans_labels = kmeans_labels.reshape(-1)
    nmi, ari, _, acc = evaluation.evaluate(img_labels, kmeans_labels)
    print("nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))

    pseudo_label = torch.from_numpy(kmeans_labels).to(device)


    last_nmi, last_ari, last_acc = 0.0, 0.0, 0.0

    for i in range(epochs):
        # print(f"----------Epoch {i+1}----------")

        # model.train()
        model.train()

        if i < args.warm_up:
            high_tuning(model, device, i)
        else:
            pair_tuning_high(model, device)

        if (i + 1) % args.eval_freq == 0:
            # print("----------Evaluation----------")
            model.eval()
            prediction_vector = []
            with torch.no_grad():
                for img, _, _ in tqdm(test_dataloader):
                    img = img.to(device)
                    logits = model(img)
                    prediction = torch.argmax(logits, dim=1)
                    prediction_vector.extend(prediction.cpu().detach().numpy())
                prediction_vector = np.array(prediction_vector)
            nmi, ari, _, acc = evaluation.evaluate(img_labels, prediction_vector)
            print("nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))
            last_nmi, last_ari, last_acc = nmi, ari, acc

    # 保存最后一个 epoch 的结果
    noise_ratio = args.noise_ratio if hasattr(args, 'noise_ratio') else 0.0
    result_dir = os.path.join(BASE_DIR, "noise", args.dataset, '')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, f"{args.criterion}_{noise_ratio}.txt")
    with open(result_file, 'a') as f:
        f.write(f"dataset={args.dataset}\tcriterion={args.criterion}\tnoise_ratio={noise_ratio}\tseed={args.seed}\tnmi={last_nmi*100:.2f}\tacc={last_acc*100:.2f}\tari={last_ari*100:.2f}\n")
    print(f"Results saved to {result_file}")

    end_time = datetime.datetime.now()
    print("结束时间：", end_time)









