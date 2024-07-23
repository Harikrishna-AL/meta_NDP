import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class EAGAttention(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_channels,
        clip_logits_min=-10,
        clip_logits_max=10,
        attn_maskout=0.1,
        attn_dropout=0.1,
        scale_degree=True,
        num_heads=8,
    ):
        super(EAGAttention, self).__init__()
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        self.clip_logits_min = clip_logits_min
        self.clip_logits_max = clip_logits_max
        self.attn_maskout = attn_maskout
        self.attn_dropout = attn_dropout
        self.scale_degree = scale_degree
        self.num_heads = num_heads

        self.queryL = nn.Linear(self.num_channels, num_channels)
        self.keyL = nn.Linear(self.num_channels, num_channels)
        self.valueL = nn.Linear(self.num_channels, num_channels)

        self.globalL = PCA(n_components=1)
        self.errorL = PCA(n_components=1)

        # self.output_nodes = nn.Linear(self.num_channels, self.num_channels)   #include if needed

    def forward(self, nodes, edges, mask=None, training=False):
        q = self.queryL(nodes)
        k = self.keyL(nodes)
        v = self.valueL(nodes)

        g = self.globalL.fit_transform(edges)
        e = self.errorL.fit_transform(edges)

        dot_dim = q.size(-1)

        # Equation (5): H_hat = clip((Q * K^T) / sqrt(d_k)) + E
        A_hat = torch.einsum("bld,bmd->blm", q, k)
        A_hat = A_hat * (dot_dim**-0.5)
        H_hat = torch.clamp(
            A_hat, self.clip_logits_min, self.clip_logits_max
        ) + torch.tensor(e, dtype=torch.float32)

        # Equation (4): A_hat = softmax(H_hat) * sigma(G)
        if mask is None:
            if self.attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(self.attn_maskout) * -1e9
                gates = torch.sigmoid(torch.tensor(g, dtype=torch.float32))
                A_tild = F.softmax(H_hat + rmask, dim=-1) * gates
            else:
                gates = torch.sigmoid(torch.tensor(g, dtype=torch.float32))
                A_tild = F.softmax(H_hat, dim=-1) * gates
        else:
            if self.attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(self.attn_maskout) * -1e9
                gates = torch.sigmoid(torch.tensor(g, dtype=torch.float32) + mask)
                A_tild = F.softmax(H_hat + mask + rmask, dim=-1) * gates
            else:
                gates = torch.sigmoid(torch.tensor(g, dtype=torch.float32) + mask)
                A_tild = F.softmax(H_hat + mask, dim=-1) * gates

        # Apply dropout to A_tild
        A_tild = F.dropout(A_tild, p=self.attn_dropout, training=training)

        # Equation (3): Attn(Q, K, V) = A_hat * V
        V_att = torch.einsum("blm,bmd->bld", A_tild, v)

        # Scale by degree if necessary
        if self.scale_degree:
            degrees = torch.sum(gates, dim=-1, keepdim=True)
            degree_scalers = torch.log1p(degrees)
            V_att = V_att * degree_scalers

        return V_att, H_hat
