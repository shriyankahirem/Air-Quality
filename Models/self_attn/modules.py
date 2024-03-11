import torch
import torch.nn as nn

from timm.layers import Mlp
from timm.models.vision_transformer import Block


class Embedding(nn.Module):
    def __init__(self, loc_dim=2, ser_dim=24, loc_embed_dim=128, ser_embed_dim=128):
        super(Embedding, self).__init__()
        self.loc_embed = Mlp(in_features=loc_dim,
                             hidden_features=loc_embed_dim * 4,
                             out_features=loc_embed_dim,)
        self.ser_embed = Mlp(in_features=ser_dim,
                             hidden_features=ser_embed_dim * 4,
                             out_features=ser_embed_dim,)
        
    def forward(self, locs, readings):
        locs = self.loc_embed(locs)
        readings = self.ser_embed(readings.transpose(1, 2))
        return locs, readings
    

class Self_Attention_Model(nn.Module):
    def __init__(self,
                 loc_dim=2, window=24,
                 dim=128, num_heads=16, depth=4,
                 qkv_bias=True, qk_norm=False, proj_drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, mlp_ratio=4.):
        super(Self_Attention_Model, self).__init__()

        self.embedding = Embedding(loc_dim=loc_dim, ser_dim=window,
                                   loc_embed_dim=dim, ser_embed_dim=dim)
        
        self.pred_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_norm=qk_norm, proj_drop=proj_drop, attn_drop=attn_drop,
                  drop_path=0.1, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        self.predict = nn.Linear(dim, 1)
        
    def forward(self, locs, readings, target_loc,):
        locs = torch.cat([locs, target_loc.unsqueeze(1)], dim=1)
        locs_embed, readings_embed = self.embedding(locs, readings)

        B, N, D = locs_embed.shape
        pred_token = self.pred_token.expand(B, -1, -1)
        readings_embed = torch.cat([readings_embed, pred_token], dim=1)

        tokens = locs_embed + readings_embed

        for blk in self.blocks:
            tokens = blk(tokens)
        
        pred = self.predict(tokens[:, -1, :])

        return pred


        return locs_embed, readings_embed