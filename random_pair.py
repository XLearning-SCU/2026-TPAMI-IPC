import os
import random
import numpy as np
import torch
from parse import get_parse
from utils.data import load_data
from our_pair_all import tsne_pair, MyPair



def sample_pairs(sample_num, labels, pair_budget):
    random_pairs = MyPair()
    while len(random_pairs) < pair_budget:
        temp1 = random.randint(0, sample_num-1)
        temp2 = random.randint(0, sample_num-1)
        while ((temp1, temp2, labels[temp1] == labels[temp2]) in random_pairs) or (temp1 == temp2):
            temp1 = random.randint(0, sample_num-1)
            temp2 = random.randint(0, sample_num-1)
        if labels[temp1] == labels[temp2]:
            random_pairs.add(temp1, temp2, 1)
        else:
            random_pairs.add(temp1, temp2, 0)

    return random_pairs



if __name__ == "__main__":
    args = get_parse()
    print(args)
    dataset_path, true_label, gpt_label, prompt = load_data(args)

    save_path = os.path.join("./data", args.dataset, '')
    img_features = np.load(save_path + args.criterion + "_image_embedding.npy")
    print(f"img_features.shape: {img_features.shape}")
    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    print(f"common_texts.shape: {common_texts.shape}")
    img_labels = np.loadtxt(save_path + args.criterion + "_labels.txt")
    img_labels = img_labels.astype(int)
    print(f"img_labels.shape: {img_labels.shape}")

    class_num = len(set(img_labels))
    print(f"class_num: {class_num}")
    sample_num = len(img_labels)
    print(f"sample_num: {sample_num}")

    common_texts = torch.from_numpy(common_texts).type(torch.float32)
    img_features = torch.from_numpy(img_features).type(torch.float32)

    new_features = img_features @ common_texts.T
    print(f"new_features.shape: {new_features.shape}")

    new_features /= torch.norm(new_features, p=2, dim=1, keepdim=True)
    # new_features = new_features - torch.mean(new_features)
    # new_features /= torch.std(new_features, dim=1, keepdim=True)
    new_features = new_features.numpy()
    random.seed(args.seed)


    if args.dataset == "fruit360" or args.dataset == "cifar10":
        args.pair_budget = 50
    else:
        args.pair_budget = 500



    selected_result = sample_pairs(sample_num, img_labels, args.pair_budget)
    # print(selected_result)

    num_p, num_n = selected_result.get_p_or_n_len()
    print(f"num_p: {num_p}, num_n: {num_n}")

    # tsne_pair(new_features, img_labels, index=None, name=str(args.seed)+'-'+args.criterion, pair=selected_result)

    pair_path = os.path.join(save_path, "random_pair", '')
    if not os.path.isdir(pair_path):
        os.makedirs(pair_path)
    save_name = str(args.pair_budget) + "-" + str(args.criterion) + "-" + str(args.seed) + ".tar"
    random_selection = {'selected_pair': selected_result}
    torch.save(random_selection, pair_path + save_name)


















