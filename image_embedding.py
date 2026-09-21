import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


import torch
import numpy as np
import torchvision
from tqdm import tqdm
from parse import get_parse
import os
import clip
from utils.data import load_data
# from utils.other_llm import data_ablation
from utils import clevr4, stanford_cars, cub
import random


if __name__ == "__main__":
    args = get_parse()
    # print(args)

    dataset_path, true_label, gpt_label, prompt = load_data(args)
    # dataset_path, true_label, gpt_label, prompt = data_ablation(args)

    # device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    # model, preprocess = clip.load('RN50', device)

    model.eval()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    if args.dataset == "clevr4":
        dataset = clevr4.Clevr4(
            root=dataset_path,
            taxonomy=args.criterion,
            transform=preprocess
        )
    elif args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=dataset_path,
            download=False,
            train=True,
            transform=preprocess
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_path,
            download=False,
            train=False,
            transform=preprocess
        )
        dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset]
        )
    elif args.dataset == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(
            root=dataset_path,
            split="train",
            transform=preprocess,
            download=False
        )
        test_dataset = torchvision.datasets.GTSRB(
            root=dataset_path,
            split="test",
            transform=preprocess,
            download=False
        )

        if args.criterion == "color":
            import json
            json_path = os.path.join(dataset_path, "gtsrb/color_gtsrb.json")
            with open(json_path) as f:
                color_data = json.load(f)
            color_labels = [item['color'] for item in color_data]
            unique_colors = sorted(set(color_labels))
            color_to_idx = {c: i for i, c in enumerate(unique_colors)}
            color_idxs = [color_to_idx[c] for c in color_labels]

            n_train = len(train_dataset)
            train_color = color_idxs[:n_train]
            test_color = color_idxs[n_train:]

            class _ColorWrapper(torch.utils.data.Dataset):
                def __init__(self, ds, labels):
                    self.ds = ds
                    self.labels = labels
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, idx):
                    img, _ = self.ds[idx]
                    return img, self.labels[idx]

            train_dataset = _ColorWrapper(train_dataset, train_color)
            test_dataset = _ColorWrapper(test_dataset, test_color)

        dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset]
        )
    elif args.dataset == "eurosat":
        dataset = torchvision.datasets.EuroSAT(
            root=dataset_path,
            transform=preprocess,
            download=False
        )
    elif args.dataset == "stanford-cars":
        csv_path = os.path.join(args.criterion, "labels.csv")
        train_dataset = stanford_cars.StanfordCarsDataset(
            root_dir=dataset_path,
            csv_path=csv_path,
            train=True,
            transform=preprocess
        )
        test_dataset = stanford_cars.StanfordCarsDataset(
            root_dir=dataset_path,
            csv_path=csv_path,
            train=False,
            transform=preprocess
        )
        dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset]
        )
    elif args.dataset == "cub" and args.criterion != "species":
        csv_path = os.path.join(args.criterion, "labels.csv")
        dataset = cub.CubDataset(
            root_dir=dataset_path,
            csv_path=csv_path,
            transform=preprocess
        )
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_path,
            transform=preprocess
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    img_features = []
    img_label = []

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            batch_features = model.encode_image(imgs)
            img_label.extend(labels.numpy())
            img_features.extend(batch_features.cpu().numpy())
    img_features = np.array(img_features)
    img_label = np.array(img_label)
    img_features /= np.linalg.norm(img_features, axis=1, keepdims=True)

    if args.dataset == "cifar10":
        if args.criterion == "scene":
            super_label = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1])
            img_label = super_label[img_label]

    if args.dataset == "eurosat":
        if args.criterion == "human":
            # 0:AnnualCrop 1:Forest 2:HerbaceousVegetation 3:Highway 4:Industrial
            # 5:Pasture 6:PermanentCrop 7:Residential 8:River 9:SeaLake
            # artificial(0): 6,0,4,7,3  natural(1): 1,2,5,8,9
            super_label = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])
            img_label = super_label[img_label]
    

    print(f"img_features.shape: {img_features.shape}")
    print(f"img_label.shape: {img_label.shape}")

    save_path = os.path.join("./data", args.dataset, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + args.criterion + "_image_embedding.npy", img_features)
    np.savetxt(save_path + args.criterion + "_labels.txt", img_label)
