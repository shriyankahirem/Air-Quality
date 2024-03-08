import torch
import torch.nn as nn

from einops import rearrange



class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class loc_embed(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64):
        super(loc_embed, self).__init__()
        self.embed = Mlp(in_features=input_dim,
                         hidden_features=embed_dim,
                         out_features=embed_dim,
                         act_layer=nn.GELU,
                         norm_layer=nn.BatchNorm1d,
                         drop=0.0)

    def forward(self, x):
        return self.embed(x)
    
    
class ser_embed(nn.Module):
    def __init__(self, in_channel=1, out_channel=64):
        super(ser_embed, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=24, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(out_channel),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        return self.embed(x)
    

class Embedding(nn.Module):
    def __init__(self, loc_dim=2, ser_dim=1, loc_embed_dim=64, ser_embed_dim=64):
        super(Embedding, self).__init__()
        self.loc_embed = loc_embed(loc_dim, loc_embed_dim)
        self.ser_embed = ser_embed(ser_dim, ser_embed_dim)

    def forward(self, x):
        """
        Input:
            x: A tuple of 3 tensors.
               The first tensor has shape (batch_size, n_sensors, 2) and contains the longitude and latitude of monitored sensors.
               The second tensor has shape (batch_size, n_steps, n_sensors) and contains the pm25 readings of monitored sensors. n_steps should be 24.
               The third tensor has shape (batch_size, 2) and contains the longitude and latitude of the target sensor.
        Output:
            A tuple of 3 tensors.
            The first tensor has shape (batch_size, n_sensors, loc_embed_dim) and contains the embedded location of monitored sensors.
            The second tensor has shape (batch_size, ser_embed_dim, n_sensors) and contains the embedded pm25 readings of monitored sensors.
            The third tensor has shape (batch_size, loc_embed_dim) and contains the embedded location of the target sensor.
        """
        loc, ser, target_loc = x
        batch_size, n_steps, n_sensors = ser.shape
        
        # embed location
        loc = rearrange(loc, 'b n d -> (b n) d', b=batch_size)   # (batch_size * n_sensors, 2)
        loc_embed = self.loc_embed(loc)   # (batch_size * n_sensors, loc_embed_dim)
        loc_embed = rearrange(loc_embed, '(b n) d -> b n d', b=batch_size)   # (batch_size, n_sensors, loc_embed_dim)
        target_loc_embed = self.loc_embed(target_loc)   # (batch_size, loc_embed_dim)

        # embed pm25 readings
        ser = rearrange(ser, 'b l n -> (b n) l').unsqueeze(1)   # (batch_size * n_sensors, 1, n_steps)
        ser_embed = self.ser_embed(ser).squeeze(-1)   # (batch_size * n_sensors, ser_embed_dim)
        ser_embed = rearrange(ser_embed, '(b n) d -> b n d', b=batch_size)   # (batch_size, n_sensors, ser_embed_dim)

        return loc_embed, ser_embed, target_loc_embed
    

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
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
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
        assert x1.shape == x2.shape, 'The shape of x1 and x2 should be the same.'
        B, N, D = x1.shape

        # q, k, v: (B, num_heads, N, head_dim)
        q = self.q_norm(self.q(x1)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(self.k(x2)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # attn: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # x: (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, D)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class Cross_Attention_Block(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm=False,
                 proj_drop=0.,
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 mlp_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Cross_Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_trop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=dim*4, out_features=dim,
                             act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)
        
    def forward(self, x1, x2):
        x = self.attn(self.norm1(x1), self.norm1(x2))
        x = x + x1
        x = x + self.mlp(self.norm2(x))
        return x


class Dual_Cross_Attention_Block(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm=False,
                 proj_drop=0.,
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 mlp_layer=Mlp):
        super().__init__()
        self.cab1 = Cross_Attention_Block(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_layer=mlp_layer
        )
        self.cab2 = Cross_Attention_Block(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_layer=mlp_layer
        )

    def forward(self, x1, x2):
        x1 = self.cab1(x1, x2)
        x2 = self.cab2(x2, x1)
        return x1, x2
    

class Geo_Attention_Model(nn.Module):
    def __init__(self, loc_dim=2, ser_dim=1, num_blocks=8,
                 dim=512, num_heads=8, qkv_bias=False, qk_norm=False, proj_drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, mlp_layer=Mlp):
        super(Geo_Attention_Model, self).__init__()
        self.embedding = Embedding(loc_dim, ser_dim, dim, dim)

        self.pred_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            Dual_Cross_Attention_Block(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer
            ) for _ in range(num_blocks)
        ])
        
        self.predict = nn.Linear(dim, 24)

    def forward(self, x):
        # loc_embed: (batch_size, n_sensors, dim)
        # ser_embed: (batch_size, n_sensors, dim)
        # target_loc_embed: (batch_size, dim)
        loc_embed, ser_embed, target_loc_embed = self.embedding(x)
        B, N, D = loc_embed.shape

        # concatenate target and monitored sensors
        loc = torch.cat((loc_embed, target_loc_embed.unsqueeze(1)), dim=1)
        ser = torch.cat((ser_embed, self.pred_token.expand(B, 1, D)), dim=1)

        # dual cross attention
        for block in self.blocks:
            loc, ser = block(loc, ser)

        # predict pm25 readings
        pred = self.predict(ser[:, -1, :])

        return pred


class Geo_Attention_Model2(nn.Module):
    def __init__(self, loc_dim=2, ser_dim=1, num_blocks=8,
                 dim=512, num_heads=8, qkv_bias=False, qk_norm=False, proj_drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, mlp_layer=Mlp):
        super(Geo_Attention_Model2, self).__init__()
        self.embedding = Embedding(loc_dim, ser_dim, dim, dim)

        self.pred_token = nn.Parameter(torch.randn(1, 1, 1))

        self.blocks = nn.ModuleList([
            Dual_Cross_Attention_Block(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer
            ) for _ in range(num_blocks)
        ])
        
        self.predict = nn.Linear(dim, 1)

    def forward(self, x):
        # loc_embed: (batch_size, n_sensors, dim)
        # ser_embed: (batch_size, n_sensors, dim)
        # target_loc_embed: (batch_size, dim)
        loc, ser, target_loc, target_reading = x
        target_reading = target_reading.unsqueeze(-1)
        target_reading[:, -1, :] = self.pred_token
        ser = torch.cat((ser, target_reading), dim=2)
        x = (loc, ser, target_loc)
        loc_embed, ser_embed, target_loc_embed = self.embedding(x)
        B, N, D = loc_embed.shape

        # concatenate target and monitored sensors
        loc = torch.cat((loc_embed, target_loc_embed.unsqueeze(1)), dim=1)
        ser = ser_embed

        # dual cross attention
        for block in self.blocks:
            loc, ser = block(loc, ser)

        # predict pm25 readings
        pred = self.predict(ser[:, -1, :])

        return pred
        
class Geo_Attention_Model3(nn.Module):
    def __init__(self, loc_dim=2, ser_dim=1, num_blocks=4,
                 dim=128, num_heads=8, qkv_bias=False, qk_norm=False, proj_drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, mlp_layer=Mlp):
        super(Geo_Attention_Model3, self).__init__()
        self.embedding = Embedding(loc_dim, ser_dim, dim, dim)

        self.pred_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            Dual_Cross_Attention_Block(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer
            ) for _ in range(num_blocks)
        ])
        
        self.predict = nn.Linear(dim, 1)

    def forward(self, x):
        # loc_embed: (batch_size, n_sensors, dim)
        # ser_embed: (batch_size, n_sensors, dim)
        # target_loc_embed: (batch_size, dim)
        loc_embed, ser_embed, target_loc_embed = self.embedding(x)
        B, N, D = loc_embed.shape

        # concatenate target and monitored sensors
        loc = torch.cat((loc_embed, target_loc_embed.unsqueeze(1)), dim=1)
        ser = torch.cat((ser_embed, self.pred_token.expand(B, 1, D)), dim=1)

        # dual cross attention
        for block in self.blocks:
            loc, ser = block(loc, ser)

        # predict pm25 readings
        pred = self.predict(ser[:, -1, :])

        return pred