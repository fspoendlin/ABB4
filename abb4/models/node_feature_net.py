import torch
from torch import nn
from torch.nn.functional import one_hot
from abb4.models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s                         # node embedding size
        self.c_pos_emb = self._cfg.c_pos_emb             # position embedding size
        self.c_timestep_emb = self._cfg.c_timestep_emb   # timestep embedding size
        
        embed_size = (  self._cfg.c_pos_emb +                                                   # pos_emb
                        1 +                                                                     # diffuse_bool
                        self._cfg.c_timestep_emb * 2 +                                          # t_trans + t_rot
                        self.c_s +                                                              # aatype 
                        (self._cfg.c_timestep_emb if self._cfg.task != 'fwd_fold' else 0) +     # t_seq 
                        (self._cfg.c_pos_emb if self._cfg.embed_imgt_numeric else 0) +          # imgt embedding
                        (14 if self._cfg.embed_region else 0 )                                  # region embedding
        )

        if self._cfg.self_condition and self._cfg.task != 'fwd_fold': # aatype_sc (logits)
            embed_size += self._cfg.aatype_pred_num_tokens

        self.aatype_embedding = nn.Embedding(21, self.c_s) # 20 AAs + <MASK>/UNK  

        # node feature embedder
        self.linear = nn.Sequential(
            nn.Linear(embed_size, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.LayerNorm(self.c_s),
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self,*,t_rot,t_trans,t_seq,
                       res_mask,diffuse_mask,
                       pos, imgt, region,
                       aatypes,aatypes_sc):
        
        # get positional embedding
        pos = pos.to(dtype=torch.float32).to(res_mask.device) # [batch, n_res]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)            # [batch, n_res, c_pos_emb]
        
        # get imgt embedding
        imgt_embed = get_index_embedding(imgt, self.c_pos_emb, max_len=2056) # distance embedding of imgt numeric part
        imgt_embed = imgt_embed * res_mask.unsqueeze(-1)      # [batch, n_res, c_pos_emb]
        
        # get region embedding
        region_embed = one_hot(region, 14)
        region_embed = region_embed * res_mask.unsqueeze(-1)  # [batch, n_res, 14]

        # append other features
        input_feats = [ pos_emb,
                        diffuse_mask[..., None],
                        self.embed_t(t_rot, res_mask),
                        self.embed_t(t_trans, res_mask),
                        self.aatype_embedding(aatypes)
                        ]
        
        input_feats += ([self.embed_t(t_seq, res_mask)] if self._cfg.task != 'fwd_fold' else [])
        input_feats += ([imgt_embed] if self._cfg.embed_imgt_numeric else [])
        input_feats += ([region_embed] if self._cfg.embed_region else [])

        if self._cfg.self_condition and self._cfg.task != 'fwd_fold':
            input_feats += ([aatypes_sc] if self._cfg.self_condition else [])

        return self.linear(torch.cat(input_feats, dim=-1))
