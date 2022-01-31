import torch.nn as nn
import torch


class DenseCL(nn.Module):

    def __init__(self, backbone_gq, backbone_gk, backbone_dq, backbone_dk, dict_size=65536, dim=128, m=0.999, tau=0.2):

        super(DenseCL, self).__init__()

        self.dim = dim
        self.m = m
        self.tau = tau

        self.encoder_gq = backbone_gq.model
        self.encoder_gk = backbone_gk.model
        self.encoder_dq = backbone_dq.model
        self.encoder_dk = backbone_dk.model

        self.encoder_gq.fc = nn.Sequential(nn.Linear(backbone_gq.dim, backbone_gq.dim), nn.ReLU(),
                                           nn.Linear(backbone_gq.dim, dim))
        self.encoder_gk.fc = nn.Sequential(nn.Linear(backbone_gq.dim, backbone_gq.dim), nn.ReLU(),
                                           nn.Linear(backbone_gq.dim, dim))

        self.encoder_dq.fc = nn.Sequential(nn.Conv2d(in_channels=backbone_gq.dim, out_channels=backbone_gq.dim,
                                                     kernel_size=(1, 1)), nn.ReLU(),
                                           nn.Conv2d(in_channels=backbone_gq.dim, out_channels=dim, kernel_size=(1, 1)))
        self.encoder_dk.fc = nn.Sequential(nn.Conv2d(in_channels=backbone_gq.dim, out_channels=backbone_gq.dim,
                                                     kernel_size=(1, 1)), nn.ReLU(),
                                           nn.Conv2d(in_channels=backbone_gq.dim, out_channels=dim, kernel_size=(1, 1)))
        for param_gq, param_gk in zip(self.encoder_gq.parameters(), self.encoder_gk.parameters()):
            param_gk.data.copy_(param_gq.data)
            param_gk.requires_grad = False

        for param_dq, param_dk in zip(self.encoder_dq.parameters(), self.encoder_dk.parameters()):
            param_dk.data.copy_(param_dq.data)
            param_dk.requires_grad = False

        self.dict_size = dict_size
        self.register_buffer("queue_g", torch.randn(self.dict_size, self.dim))
        self.queue_g = nn.functional.normalize(self.queue_g, dim=0)

        self.register_buffer("queue_d", torch.randn(self.dict_size, self.dim))
        self.queue_d = nn.functional.normalize(self.queue_d, dim=0)

        self.ptr = 0

    def momentum_update_key_encoders(self):
        for param_gq, param_gk in zip(self.encoder_gq.parameters(), self.encoder_gk.parameters()):
            param_gk.data = param_gk.data * self.m + param_gq.data * (1. - self.m)
        for param_dq, param_dk in zip(self.encoder_dq.parameters(), self.encoder_dk.parameters()):
            param_dk.data = param_dk.data * self.m + param_dq.data * (1. - self.m)

    def update_queues(self, k_g, k_d):
        batch_size = k_g.shape[0]
        self.queue_d[self.ptr: self.ptr + batch_size] = k_d
        self.queue_g[self.ptr: self.ptr + batch_size] = k_g
        self.ptr = (self.ptr + batch_size) % self.dict_size

    def forward(self, in_q, in_k):
        q_g = self.encoder_gq(in_q)
        q_d = self.encoder_dq(in_q)
        with torch.no_grad():
            self.momentum_update_key_encoders()
            k_g = self.encoder_gk(in_k)
            k_d = self.encoder_dk(in_k)

        self.update_queues(k_g, k_d)
