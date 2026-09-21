import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import torch
import numpy as np
import torchvision
from tqdm import tqdm
import clip
import sys
from utils.data import load_data
# from utils.other_llm import data_ablation
from parse import get_parse
import random


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    args = get_parse()
    # print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    # model, preprocess = clip.load('RN50', device)

    model.eval()

    dataset_path, _, gpt_label, prompt = load_data(args)
    # dataset_path, true_label, gpt_label, prompt = data_ablation(args)
    # print(f"prompt: {prompt}")
    # print(f"gpt_labels: {gpt_label}")
    # print(f"len(gpt_label): {len(gpt_label)}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    gpt_texts = []
    only_texts = []

    with torch.no_grad():
        for j in tqdm(gpt_label):
            gpt_word_with_prompt = prompt + j
            gpt_batch_inputs = clip.tokenize(gpt_word_with_prompt).to(device)
            gpt_batch = model.encode_text(gpt_batch_inputs)
            gpt_texts.extend(gpt_batch.cpu().numpy())
    gpt_texts = np.array(gpt_texts)
    gpt_texts /= np.linalg.norm(gpt_texts, axis=1, keepdims=True)
    print(f"gpt_texts.shape: {gpt_texts.shape}")

    # with torch.no_grad():
    #     for j in tqdm(gpt_label):
    #         gpt_batch_inputs = clip.tokenize(j).to(device)
    #         gpt_batch = model.encode_text(gpt_batch_inputs)
    #         only_texts.extend(gpt_batch.cpu().numpy())
    # only_texts = np.array(only_texts)
    # only_texts /= np.linalg.norm(only_texts, axis=1, keepdims=True)
    # print(f"only_texts.shape: {only_texts.shape}")

    save_path = os.path.join(BASE_DIR, "data", args.dataset, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # np.save(save_path + args.criterion + "_gpt_texts.npy", gpt_texts)
    # np.save(save_path + args.criterion + "_only_texts.npy", only_texts)
    np.save(save_path + args.criterion + "_common_texts.npy", gpt_texts)
    # np.save(save_path + args.criterion + "_llama_texts.npy", gpt_texts)
    # np.save(save_path + args.criterion + "_qwen_texts.npy", gpt_texts)










