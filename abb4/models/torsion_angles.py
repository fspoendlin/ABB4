import torch
from torch import nn
from torch.nn import functional as F


class TorsionAngles(nn.Module):

    def __init__(self, module_cfg):
        super().__init__()
        self._cfg = module_cfg
        c_hidden = self._cfg.c_hidden

        self.linear_in = nn.Linear(self._cfg.c_in, c_hidden)
        self.linear_initial = nn.Linear(self._cfg.c_in, c_hidden)

        self.residual1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden)
        )

        self.residual2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden)
        )

        self.final_pred = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_hidden, self._cfg.no_angles * 2)
        )

    def forward(self, node_feats, node_feats_initial):
        node_feats = self.linear_in(F.relu(node_feats))                      # [batch, n_res, c_hidden]
        node_feats_initial = self.linear_initial(F.relu(node_feats_initial)) # [batch, n_res, c_hidden]
        node_feats = node_feats + node_feats_initial

        torsions = node_feats + self.residual1(node_feats)
        torsions = torsions + self.residual2(torsions)
        torsions = self.final_pred(torsions)

        torsions = torsions.view(torsions.shape[:-1] + (-1, 2))              # [batch, n_res, no_angles, 2]

        norm_denom = torch.sqrt(                                             # [batch, n_res, no_angles, 1] 
                        torch.sum(torsions ** 2, dim=-1, keepdim=True)
                    )
        torsions = torsions / norm_denom

        return torsions, norm_denom
