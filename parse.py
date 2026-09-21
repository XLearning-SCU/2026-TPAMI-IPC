import argparse


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--data_path', type=str, default="/opt/data/private/honglin/", help='data path')
    parser.add_argument('--dataset', type=str, default="eurosat", help='clevr4, cards, stanford-cars, cub, cifar10, fruit360, gtsrb, eurosat')
    parser.add_argument('--criterion', type=str, default="scene", help='clustering criterion')
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument('--VLM', type=str, default="CLIP", help='VLM model')
    parser.add_argument('--strategy', type=str, default="rep+div+hard", help='selection strategy')
    parser.add_argument('--random', type=int, default=0, help='0:ours 1:random')
    parser.add_argument('--high_threshold', type=float, default=0.9, help='high confidence threshold')
    parser.add_argument('--pair_budget', type=int, default=500, help='number of selected pairs')
    parser.add_argument('--each_num', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='idc learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--warm_up', type=int, default=10, help='number of warm_up epochs')
    parser.add_argument('--eval_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--noise_ratio', type=float, default=0., help='')
    args = parser.parse_args()
    return args
