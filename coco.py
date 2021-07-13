import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle

attributes = [225, 74, 778, 167, 97, 789, 19, 23, 46, 285, 602, 797, 798, 805, 806, 96, 88, 194, 807, 808, 809, 815, 819, 822, 826, 828, 282]

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx



class Thyroid(data.Dataset):
    def __init__(self, root, transform=None, phase='train', vocab=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        self.vocab = vocab

        self.get_anno()
        self.num_classes = len(attributes)

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))

    def gen_A(self):
        adj = np.zeros((self.num_classes, self.num_classes), np.float32)
        feat_list = []
        for c, i in enumerate(self.features):
            if i != '0':
                feat_list.append(c)
        for i in feat_list:
            for j in feat_list:
                adj[i][j] = 1
        return adj


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        vocab = self.vocab
        filename = item['filename']
        self.features = item['features']
        tokens = item['captions'].split()
        caption = []
        caption.append(vocab['<start>'])
        caption.extend([vocab[token] for token in tokens])
        caption.append(vocab['<end>'])
        target = torch.LongTensor(caption)
        length = torch.LongTensor([len(caption)])

        img = Image.open(os.path.join(self.root, 'ThyroidImage2021', filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        self.adj = self.gen_A()
        adj = torch.Tensor(self.adj)
        # print('[data/img]:',img)

        return img, adj, target, length


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, adjs, captions, cap_lens = zip(*data)
    img = torch.stack(images)
    adj = torch.stack(adjs)
    # cap_lens = torch.stack(cap_lens)
    targets = torch.zeros(len(captions), max(cap_lens)).long()
    for i, cap in enumerate(captions):
        end = cap_lens[i]
        targets[i, :end] = cap[:end]
    return img, adj, targets, cap_lens