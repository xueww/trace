import torch
import torch.nn as nn

from ultimate_dti import UltimateDTI, WideAndDeepHead


class TraceDTIInferenceModel(UltimateDTI):
    def __init__(
        self,
        prot_channels,
        drug_channels,
        hgnn_dim,
        embed_dim=256,
        mlp_hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__(
            prot_channels,
            drug_channels,
            hgnn_dim,
            embed_dim,
            mlp_hidden_dim,
            dropout,
            use_hgnn=False,
            use_bilinear=True,
        )
        del self.cross_attn_reverse
        del self.rev_attn_pool
        self.prot_scconv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.drug_scconv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        pooled_dim = embed_dim * 3
        self.int_norm = nn.LayerNorm(pooled_dim)
        self.mlp = WideAndDeepHead(
            in_features=pooled_dim * 3,
            hidden_dims=[mlp_hidden_dim, mlp_hidden_dim // 2, 128],
            dropout_p=dropout,
        )

    def forward(self, prot_feat, drug_seq_feat, drug_graph_feat):
        prot_mask = self._make_mask(prot_feat, dim=1)
        drug_mask = self._make_mask(drug_seq_feat, dim=1)
        prot = self.prot_scconv(self.prot_proj(prot_feat)).transpose(1, 2)
        drug = self.drug_scconv(self.drug_proj(drug_seq_feat)).transpose(1, 2)
        prot_interaction = self.cross_attn_prot(drug, prot, drug_mask, prot_mask)
        drug_interaction = self.cross_attn_drug(drug, drug, drug_mask, drug_mask)
        prot_pooled = self._hybrid_pool(
            prot_interaction, prot_mask, self.prot_attn_pool
        )
        drug_pooled = self._hybrid_pool(
            drug_interaction, drug_mask, self.drug_attn_pool
        )
        interaction = self.int_norm(drug_pooled * prot_pooled)
        score = self.mlp(
            torch.cat([drug_pooled, prot_pooled, interaction], dim=-1)
        )
        return score, prot_pooled, drug_pooled
