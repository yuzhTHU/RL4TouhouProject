import random
import numpy as np
import torch
import argparse
from datetime import datetime


def get_args(**kwargs):
    parser = argparse.ArgumentParser()

    # 通用参数
    parser.add_argument("--UUID", type=str, default="testproj")
    parser.add_argument("--SAVE_DIRECTORY", type=str, default="PPO")
    parser.add_argument("--LOAD_MODEL", type=str, default=None)
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--DEVICE", type=str, default="cpu")
    parser.add_argument("--MEMORY_CAPACITY", type=int, default=10000)
    parser.add_argument("--MODEL", type=str, default="PPO")
    parser.add_argument("--EPISODE", type=int, default=2000)
    parser.add_argument("--MAX_EP", type=int, default=20000)


    # 强化学习模型公用参数
    parser.add_argument("--F_DIM", type=int, default=256)
    parser.add_argument("--A_DIM", type=int, default=7)
    parser.add_argument("--FIGSIZE", type=int, default=128)
    parser.add_argument("--GAMMA", type=float, default=0.99)
    parser.add_argument("--LR_0", type=float, default=1e-4)
    parser.add_argument("--LR_A", type=float, default=1e-4)
    parser.add_argument("--LR_C", type=float, default=1e-3)

    # SAC 参数
    parser.add_argument("--TAU", type=float, default=1e-3)
    parser.add_argument("--ALPHA", type=float, default=0.2)
    parser.add_argument("--LOG_STD_MIN", type=float, default=-20)
    parser.add_argument("--LOG_STD_MAX", type=float, default=2)
    parser.add_argument("--BATCH_SIZE", type=int, default=32)
    parser.add_argument("--RHO_START", type=float, default=0.5)
    parser.add_argument("--RHO_END", type=float, default=0.2)
    parser.add_argument("--RHO_DEC", type=float, default=3e-3)
    parser.add_argument("--VAR_V", type=float, default=0.11)
    parser.add_argument("--VAR_W", type=float, default=0.10)

    # 自定义参数
    for key, value in kwargs.items():
        parser.add_argument(f"--{key}", **value)
    args, unknown = parser.parse_known_args()
    if len(unknown):
        print(unknown)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_time():
    datetime.now().strftime('%Y%b%d_%H%M')