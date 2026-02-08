import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from tqdm import tqdm
import numpy as np


def normalize(x):
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
    return x

def prototype_generation(reserve_vector_count, num_features):
    """生成伪目标向量组"""
    points = torch.randn(reserve_vector_count, num_features)
    points = F.normalize(points, dim=-1)
    points = torch.nn.Parameter(points)

    opt = torch.optim.Adam([points], lr=0.001)

    tqdm_gen = tqdm(range(10))

    for _ in tqdm_gen:
        # Compute the cosine similarity.
        sim = F.cosine_similarity(points[None, :, :], points[:, None, :], dim=-1)
        l = torch.log(torch.exp(sim / 1.0).sum(dim=1)).sum() / points.shape[0]
        l.backward()
        opt.step()
        points.data = F.normalize(points.data)

    # Setting Reserved vectors
    return points.data


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f1 = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f1.append(module)
        # encoder
        self.f1 = nn.Sequential(*self.f1)
        # projection head
        self.g1 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True), nn.BatchNorm1d(128))
        self.h1 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True), nn.BatchNorm1d(128))

        self.f2 = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f2.append(module)
        # encoder
        self.f2 = nn.Sequential(*self.f2)
        # projection head
        self.g2 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True), nn.BatchNorm1d(128))
        self.h2 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True), nn.BatchNorm1d(128))

        self.prototypes1 = nn.Parameter(prototype_generation(16, feature_dim))
        self.prototypes2 = nn.Parameter(prototype_generation(16, feature_dim))
        self.prototypes3 = nn.Parameter(prototype_generation(16, feature_dim))

    def forward(self, ms, pan):
        # encoder
        ms = self.f1(ms)
        feature_ms = torch.flatten(ms, start_dim=1)
        pan = self.f2(pan)
        feature_pan = torch.flatten(pan, start_dim=1)

        out_ms1 = self.g1(feature_ms)
        out_ms2 = self.h1(feature_ms)
        out_pan1 = self.g2(feature_pan)
        out_pan2 = self.h2(feature_pan)

        return F.normalize(out_ms1, dim=-1), F.normalize(out_ms2, dim=-1), F.normalize(out_pan1, dim=-1), F.normalize(out_pan2, dim=-1)






