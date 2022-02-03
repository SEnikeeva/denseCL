import torch.nn as nn
import torch


class DenseCL(nn.Module):

    def __init__(self, backbone_q, backbone_k, dict_size=65536, dim=128, m=0.999, tau=0.2):

        super(DenseCL, self).__init__()

        self.dim = dim
        self.m = m
        self.tau = tau

        self.encoder_q = backbone_q
        self.encoder_k = backbone_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.dict_size = dict_size

        # dim x dict_size for multiplication
        self.register_buffer("queue_g", torch.randn(dim, self.dict_size))
        self.register_buffer("queue_d", torch.randn(dim, self.dict_size))
        self.queue_g = nn.functional.normalize(self.queue_g, dim=0)
        self.queue_d = nn.functional.normalize(self.queue_d, dim=0)

        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoders(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def update_queues(self, k_g, k_d):
        # batch_size = k_g.shape[0]
        # self.queue_g[self.ptr: self.ptr + batch_size] = k_g
        # self.queue_d[self.ptr: self.ptr + batch_size] = k_d
        # self.ptr = (self.ptr + batch_size) % self.dict_size

        k_d = nn.functional.normalize(k_d.mean(dim=2), dim=1)
        batch_size = k_g.shape[0]

        ptr = int(self.ptr)

        self.queue_g[:, ptr:ptr + batch_size] = k_g.T
        self.queue_d[:, ptr:ptr + batch_size] = k_d.T
        ptr = (ptr + batch_size) % self.dict_size  # move pointer

        self.ptr[0] = ptr

    def forward(self, in_q, in_k):

        bs = in_k.size(0)
        g_q, d_q, feat_q = self.encoder_q(in_q)
        d_q = d_q.view(bs, d_q.size(1), -1)
        feat_q = feat_q.view(bs, feat_q.size(1), -1)

        with torch.no_grad():

            self.momentum_update_key_encoders()
            g_k, d_k, feat_k = self.encoder_k(in_k)
            d_k = d_k.view(bs, d_k.size(1), -1)
            feat_k = feat_k.view(bs, feat_k.size(1), -1)
            cosine = torch.einsum('bqi,bqj->bij', feat_k, feat_q)
            match_idx = cosine.argmax(dim=1)
            d_q = d_q.gather(2, match_idx.unsqueeze(1).expand(-1, self.dim, -1))

        input_pos_g = torch.einsum('bd,bd->b', [g_q, g_k]).view(bs, 1)
        input_neg_g = torch.einsum('bd,dq->bq', [g_q, self.queue_g.clone().detach()])
        input_g = torch.concat((input_pos_g, input_neg_g), dim=1) / self.tau
        labels_g = torch.zeros(input_g.size(0), dtype=torch.long)

        input_pos_d = torch.einsum('bdz,bdz->bz', [d_k, d_q]).view(bs, 1, d_q.size(-1))
        input_neg_d = torch.einsum('bdz,dq->bqz', [d_q, self.queue_d.clone().detach()])
        input_d = torch.concat((input_pos_d, input_neg_d), dim=1) / self.tau
        labels_d = torch.zeros((input_d.size(0), input_d.size(-1)), dtype=torch.long)

        self.update_queues(g_k, d_k)
        return input_g, labels_g, input_d, labels_d

    # def forward(self, im_q, im_k):
    #     """
    #     Input:
    #         im_q: a batch of query images
    #         im_k: a batch of key images
    #     Output:
    #         logits, targets
    #     """
    #
    #     # compute query features
    #     q, dense_q, feat_q = self.encoder_q(im_q)  # queries: NxC
    #     q = nn.functional.normalize(q, dim=1)
    #     n, c, h, w = feat_q.size()
    #     dim_dense = dense_q.size(1)
    #     dense_q, feat_q = dense_q.view(n, dim_dense, -1), feat_q.view(n, c, -1)
    #     dense_q = nn.functional.normalize(dense_q, dim=1)
    #
    #     # compute key features
    #     with torch.no_grad():  # no gradient to keys
    #         self.momentum_update_key_encoders()  # update the key encoder
    #
    #         # # shuffle for making use of BN
    #         # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
    #
    #         k, dense_k, feat_k = self.encoder_k(im_k)  # keys: NxC
    #         k = nn.functional.normalize(k, dim=1)
    #         dense_k, feat_k = dense_k.view(n, dim_dense, -1), feat_k.view(n, c, -1)
    #         dense_k_norm = nn.functional.normalize(dense_k, dim=1)
    #
    #         # # undo shuffle
    #         # k, feat_k, dense_k_norm = self._batch_unshuffle_ddp(
    #         #         k, feat_k, dense_k_norm, idx_unshuffle)
    #
    #         ## match
    #         feat_q_norm = nn.functional.normalize(feat_q, dim=1)
    #         feat_k_norm = nn.functional.normalize(feat_k, dim=1)
    #         cosine = torch.einsum('nca,ncb->nab', feat_q_norm, feat_k_norm)
    #         pos_idx = cosine.argmax(dim=-1)
    #         dense_k_norm = dense_k_norm.gather(2, pos_idx.unsqueeze(1).expand(-1, dim_dense, -1))
    #
    #     # compute logits
    #     # Einstein sum is more intuitive
    #     # positive logits: Nx1
    #     l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    #     # negative logits: NxK
    #     l_neg = torch.einsum('nc,ck->nk', [q, self.queue_g.clone().detach()])
    #
    #     # logits: Nx(1+K)
    #     logits = torch.cat([l_pos, l_neg], dim=1)
    #
    #     # apply temperature
    #     logits /= self.tau
    #
    #     # labels: positive key indicators
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long)
    #
    #
    #     ## densecl logits
    #     d_pos = torch.einsum('ncm,ncm->nm', dense_q, dense_k_norm).unsqueeze(1)
    #     d_neg = torch.einsum('ncm,ck->nkm', dense_q, self.queue_d.clone().detach())
    #     logits_dense = torch.cat([d_pos, d_neg], dim=1)
    #     logits_dense = logits_dense / self.tau
    #     labels_dense = torch.zeros((n, h*w), dtype=torch.long)
    #
    #     # dequeue and enqueue
    #     self.update_queues(k, dense_k)
    #
    #     return logits, labels, logits_dense, labels_dense


