import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.layers import Mlp
from einops import rearrange


class Loc_Embedding(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, ratio=4.0):
        super(Loc_Embedding, self).__init__()
        self.embed = Mlp(in_features=input_dim,
                         hidden_features=int(embed_dim * ratio),
                         out_features=embed_dim,)
        
    def forward(self, x):
        return self.embed(x)
    

class Series_Embedding(nn.Module):
    def __init__(self, input_channel=1, out_channel=64, ratio=4.0, window=24):
        super(Series_Embedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=int(out_channel * ratio),
                      kernel_size=window, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(int(out_channel * ratio)),
            nn.Conv1d(in_channels=int(out_channel * ratio), out_channels=out_channel,
                      kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        return self.embed(x)
    

class Embedding(nn.Module):
    def __init__(self, loc_dim=2, series_dim=1, dim=64, ratio=4.0, window=24):
        super(Embedding, self).__init__()
        self.loc_embed = Loc_Embedding(input_dim=loc_dim, embed_dim=dim, ratio=ratio)
        self.series_embed = Series_Embedding(input_channel=series_dim, out_channel=dim,
                                             ratio=ratio, window=window)

    def forward(self, x):
        """
        Input:
            A tuple of two tensors, (locs, readings)
            locs: a tensor with shape (batch_size, n_sensors, 2)
            readings: a tensor with shape (batch_size, window, n_sensors)
        Output:
            A tuple of two tensors.
            loc_embed: a tensor with shape (batch_size, n_sensors, dim)
            readings_embed: a tensor with shape (batch_size, n_sensors, dim)
        """
        locs, readings = x
        batch_size, window, n_sensors = readings.shape

        # embed location
        locs = rearrange(locs, 'b n d -> (b n) d', b=batch_size)   # (batch_size * n_sensors, 2)
        loc_embed = self.loc_embed(locs)   # (batch_size * n_sensors, dim)
        loc_embed = rearrange(loc_embed, '(b n) d -> b n d', b=batch_size)   # (batch_size, n_sensors, dim)

        # embed pm25 readings
        readings = rearrange(readings, 'b w n -> (b n) w', b=batch_size).unsqueeze(1)   # (batch_size * n_sensors, 1, window)
        readings_embed = self.series_embed(readings).squeeze()   # (batch_size * n_sensors, dim)
        readings_embed = rearrange(readings_embed, '(b n) d -> b n d', b=batch_size)   # (batch_size, n_sensors, dim)

        return loc_embed, readings_embed
    

class Cross_Attention(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_layer=nn.LayerNorm):
        super(Cross_Attention, self).__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, 'The shape of x1 and x2 should be the same'
        B, N, D = x1.shape

        # q, k, v: (B, num_heads, N, head_dim)
        q = self.q_norm(self.q(x1)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(self.k(x2)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # attn: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x: (B, num_heads, N, head_dim) ->
        #    (B, N, num_heads, head_dim) ->
        #    (B, N, D)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))

        return x
    

class Cross_Attention_Block(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=8,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_norm=False,
                 proj_drop=0.,
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 mlp_layer=Mlp,
                 fix_context=True):
        super(Cross_Attention_Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Cross_Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             out_features=dim,
                             act_layer=act_layer,
                             norm_layer=norm_layer,
                             drop=proj_drop)
        
        self.fix_context = fix_context
        
    def forward(self, x1, x2):
        x = x1 + self.attn(self.norm1(x1), self.norm1(x2))
        x = x + self.mlp(self.norm2(x))

        if self.fix_context:
            return x, x2
        else:
            return x1, x 

class MaskedAutoEncoder(nn.Module):
    def __init__(self,
                 loc_dim=2, series_dim=1, window=24,
                 dim=128, num_heads=16, depth=4,
                 decoder_dim=64, decoder_num_heads=8, decoder_depth=2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MaskedAutoEncoder, self).__init__()

        # -----------------------------------------------------------------
        # MAE Encoder
        self.embedding = Embedding(loc_dim=loc_dim, series_dim=series_dim,
                                   dim=dim, ratio=mlp_ratio, window=window)
        
        self.blocks = nn.ModuleList([
            Cross_Attention_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=True, qk_norm=True, norm_layer=norm_layer, fix_context=False)
            for _ in range(depth)
        ])
        self.norm = norm_layer(dim)
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # MAE Decoder
        self.decoder_loc_embedding = nn.Linear(dim, decoder_dim, bias=True)
        self.decoder_ser_embedding = nn.Linear(dim, decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.decoder_blocks = nn.ModuleList([
            Cross_Attention_Block(dim=decoder_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=True, qk_norm=True, norm_layer=norm_layer, fix_context=True)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, 1, bias=True)



    def random_masking(self, x1, x2, mask_ratio):
        """
        Randomly mask a portion of sensors' readings.
        Random sample is done by argsort random noise
        """
        B, N, D = x1.shape
        len_keep = int(N * (1 - mask_ratio))

        # sort noise for each sample
        noise = torch.rand(B, N, device=x1.device)   # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)   # ascending order, small noise is keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first len_keep sensors
        ids_keep = ids_shuffle[:, :len_keep]
        x1_masked = torch.gather(x1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x2_masked = torch.gather(x2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x1.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask in correct order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x1_masked, x2_masked, mask, ids_restore

    def forward_encoder(self, loc_embed, readings_embed):
        """
        Input:
            loc_embed: a tensor with shape (batch_size, n_sensors, dim)
            readings_embed: a tensor with shape (batch_size, n_sensors, dim)
        Output:
            loc_embed_masked: a tensor with shape (batch_size, n_sensors*(1-mask_ratio), dim)
            readings_embed_masked: a tensor with shape (batch_size, n_sensors*(1-mask_ratio), dim)
            mask: a tensor with shape (batch_size, n_sensors), n_sensors are unshuffled
            ids_restore: a tensor with shape (batch_size, n_sensors), used to restore the original order
        """
        # random masking
        loc_embed_masked, readings_embed_masked, mask, ids_restore = self.random_masking(loc_embed, readings_embed, mask_ratio=0.25)

        # apply encoder blocks
        for blk in self.blocks:
            loc_embed_masked, readings_embed_masked = blk(loc_embed_masked, readings_embed_masked)
        loc_embed_masked = self.norm(loc_embed_masked)
        readings_embed_masked = self.norm(readings_embed_masked)

        return loc_embed_masked, readings_embed_masked, mask, ids_restore

    def forward_decoder(self, loc_embed, readings_embed_masked, ids_restore):
        """
        Input:
            loc_embed: a tensor with shape (batch_size, n_sensors, dim), sensors are unshuffled
            readings_embed_masked: a tensor with shape (batch_size, n_sensors*(1-mask_ratio), dim)
            ids_restore: a tensor with shape (batch_size, n_sensors), used to restore the original order
        Output:
            readings: a tensor with shape (batch_size, n_sensors)
        """
        B, N, D = loc_embed.shape
        N_KEEP = readings_embed_masked.shape[1]

        # embed location
        loc_embed = self.decoder_loc_embedding(loc_embed)
        # embed readings
        mask_tokens = self.mask_token.repeat(B, N - N_KEEP, 1)
        readings_embed_masked = self.decoder_ser_embedding(readings_embed_masked)
        readings_embed = torch.cat([readings_embed_masked, mask_tokens], dim=1)
        readings_embed = torch.gather(readings_embed, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, loc_embed.shape[-1]))   # unshuffle sensors

        # apply decoder blocks
        for blk in self.decoder_blocks:
            loc_embed, readings_embed = blk(loc_embed, readings_embed)
        readings_embed = self.decoder_norm(readings_embed)

        # predict pm25 readings
        readings = self.decoder_pred(readings_embed).squeeze()

        return readings

    def forward_loss(self, readings, readings_pred, mask):
        """
        Input:
            readings: a tensor with shape (batch_size, window, n_sensor)
            readings_pred: a tensor with shape (batch_size, n_sensor)
            mask: a tensor with shape (batch_size, n_sensor)
        Output:
            loss: a scalar
        """
        target = readings[:, -1, :]   # (batch_size, n_sensors)
        loss = (target - readings_pred) ** 2

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x):
        # embedding
        # (batch_size, n_sensors, dim), (batch_size, n_sensors, dim)
        loc_embed, readings_embed = self.embedding(x)

        # encoder
        loc_embed_masked, readings_embed_masked, mask, ids_restore = self.forward_encoder(loc_embed, readings_embed)

        # decoder
        readings_pred = self.forward_decoder(loc_embed, readings_embed_masked, ids_restore)

        # loss
        loss = self.forward_loss(x[1], readings_pred, mask)

        return loss, readings_pred, mask