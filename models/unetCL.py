import torch.nn as nn
import torch
from torch.nn import functional as f


class UNetCL(nn.Module):

    def __init__(self, backbone_q, backbone_k, dict_size=65536, dim=128, m=0.999, tau=0.2, is_cuda=True):

        super(UNetCL, self).__init__()

        self.dim = dim
        self.m = m
        self.tau = tau
        self.is_cuda = is_cuda

        self.encoder_q = backbone_q
        self.encoder_k = backbone_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.dict_size = dict_size

        # dim x dict_size for multiplication
        self.register_buffer("queue", torch.randn(dim, self.dict_size))
        self.queue = f.normalize(self.queue, dim=0)

        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoders(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def update_queues(self, k):
        k = f.normalize(k.mean(dim=2), dim=1)  # global average pooling
        batch_size = k.shape[0]

        ptr = int(self.ptr)
        if ptr + batch_size > self.dict_size:
            part_size = self.dict_size - ptr
            self.queue[:, ptr:] = k.T[:, :part_size]
            self.queue[:, :batch_size - part_size] = k.T[:, part_size:]
            ptr = batch_size - part_size
        else:
            self.queue[:, ptr:ptr + batch_size] = k.T
            ptr = (ptr + batch_size) % self.dict_size  # move pointer

        self.ptr[0] = ptr

    # TODO: add shuffle
    def forward(self, in_q, in_k):
        bs = in_k.size(0)
        q = self.encoder_q(in_q)
        q = f.normalize(q, dim=1)  # l2 normalized feature vector
        q = q.view(bs, q.size(1), -1)
        with torch.no_grad():
            self.momentum_update_key_encoders()
            k = self.encoder_k(in_k)
            feat_k = k.clone()
            k = k.view(bs, k.size(1), -1)
            feat_k = feat_k.view(bs, feat_k.size(1), -1)
            k_norm = f.normalize(k, dim=1)  # l2 normalized feature vector
            feat_k_norm = nn.functional.normalize(feat_k, dim=1)
            cosine = torch.einsum('nca,ncb->nab', q, feat_k_norm)
            match_idx = cosine.argmax(dim=-1)
            k_norm = k_norm.gather(2, match_idx.unsqueeze(1).expand(-1, self.dim, -1))

        output_pos = torch.einsum('bdz,bdz->bz', [k_norm, q]).view(bs, 1, q.size(-1))
        output_neg = torch.einsum('bdz,dq->bqz', [q, self.queue.clone().detach()])
        output = torch.concat((output_pos, output_neg), dim=1) / self.tau
        if self.is_cuda:
            target = torch.zeros((output.size(0), output.size(-1)), dtype=torch.long).cuda()
        else:
            target = torch.zeros((output.size(0), output.size(-1)), dtype=torch.long)
        self.update_queues(k)
        return output, target
