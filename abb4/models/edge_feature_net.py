import torch
from torch import nn

from abb4.models.utils import get_index_embedding, calc_distogram

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins + 2
        if self._cfg.self_condition:
            total_edge_feats += (self._cfg.num_bins)

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos):
        # AlphaFold 2 Algorithm 4 & 5
        # based on OpenFold utils/tensor_utils.py
        # pos: [b, n_res]
        d = pos[:, :, None] - pos[:, None, :]   # [b, n_res, n_res]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, node_embed, trans_t, trans_sc, edge_mask, diffuse_mask, res_idx):
        num_batch, num_res, _ = node_embed.shape
        p_i = self.linear_s_p(node_embed)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)

        pos = res_idx.to(node_embed.device)
        relpos_feats = self.embed_relpos(pos)

        dist_feats = calc_distogram(
            trans_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(
            trans_sc, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, diff_feat]
        if self._cfg.self_condition:
            all_edge_feats += [sc_feats]

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= edge_mask.unsqueeze(-1)
        return edge_feats
