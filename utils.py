from datetime import datetime
import os
import logging
from scipy.sparse import coo_matrix
from setting import SetParameter
import torch
from math import radians, cos, sin, asin, sqrt
import numpy as np
import random


args = SetParameter()


# 打印和保存
class Logger:
    def __init__(self, file):
        # 防止重复打印
        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)
        # filemode
        # 'w'：覆盖写入模式，在每次写入日志时，都会清空原来的内容并重新写入。
        # 'a'：追加写入模式，在每次写入日志时，会在文件末尾继续写入，不会删除原来的内容。
        self.logger = logging.getLogger()
        if not logging.root.hasHandlers():
            logging.basicConfig(level=logging.DEBUG,
                                filename=os.path.join(file, "console.log"),
                                filemode='w',
                                format='')
            console = logging.StreamHandler()  # 创建一个控制台处理器，用于将日志消息输出到控制台。
            console.setLevel(logging.DEBUG)  # 设置控制台处理器的日志级别
            self.logger.addHandler(console)

    def print_log(self):
        return self.logger


def calculate_time_slot(date_time):
    # format = '%Y-%m-%d %H:%M:%S'
    format = '%Y-%m-%dT%H:%M:%SZ'
    date_time = datetime.strptime(date_time, format)

    day_of_week = date_time.weekday()  # 获取星期几,0代表星期一
    hour = date_time.hour
    minute = date_time.minute

    slot_of_day = int(hour * 2 + minute / 30)
    slot_of_week = day_of_week * 48 + slot_of_day

    out = int(slot_of_week)
    return out


def generate_src_mask(n, num_heads, s):
    # Generate square subsequent mask
    mask = torch.triu(torch.full((s, s), float('-inf')), diagonal=1)
    # Repeat mask N*num_heads times and reshape to (N*num_heads, S, S)
    src_mask = mask.repeat(n * num_heads, 1, 1).view(n * num_heads, s, s)
    return src_mask


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return graph


def deduplicate_matrix(tgt_m, src_m):
    mask = torch.any(torch.eq(tgt_m.unsqueeze(-2), src_m.unsqueeze(-1)), dim=1)
    tgt_m[mask] = -1
    sequence = torch.arange(tgt_m.size(1) - 1, -1, -1)
    matrix = sequence.unsqueeze(0).repeat(tgt_m.size(0), 1)
    mask = (tgt_m == -1)
    matrix[mask] = -1
    sorted_vals, sorted_indices = torch.sort(matrix, dim=1, descending=True)
    out = tgt_m[torch.arange(tgt_m.size(0)).unsqueeze(-1), sorted_indices]
    return out
