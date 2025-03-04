import torch
import torch.nn as nn
from tools import utils 
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from point_transformer_pytorch import PointTransformerLayer

class PointTransformerEncoder(nn.Module):
    def __init__(self, depth=1):
        self.blocks = nn.ModuleList([
            PointTransformerLayer(
                dim=256,
                pos_mlp_hidden_dim=256,
                attn_mlp_hidden_mult=1)
            for i in range(depth)])
    
    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x, pos)
        return x


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G K 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)  
        feature = self.second_conv(feature) 
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] 
        return feature_global.reshape(bs, g, self.encoder_channel) 

class MaskTransformer(nn.Module):
    def __init__(self, config, mask_ratio):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = mask_ratio
        self.trans_dim = config['model']['trans_dim']
        self.depth = config['model']['depth'] 
        self.drop_path_rate = config['model']['drop_path_rate']
        self.num_heads = config['model']['num_heads'] 

        # embedding
        self.encoder_dims =  config['model']['encoder_dims']
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config['model']['mask_type']

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.pointtransformer = (
            PointTransformerLayer(
                dim=384,
                pos_mlp_hidden_dim=512,
                attn_mlp_hidden_mult=2,
            )
        )
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_test(self, center, idxy=0, noaug=False):
        '''
            center : B G 3 
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.zeros(G)
            if(((self.num_mask*idxy)%G)+self.num_mask<G):
                mask[(self.num_mask*idxy)%G : ((self.num_mask*idxy)%G)+self.num_mask] = 1
            else:
                mask[(self.num_mask*idxy)%G: ] = 1,
                mask[0: (self.num_mask*(idxy+1))%G] = 1
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) 

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3 
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) 

    def forward(self, neighborhood, center, noaug=False, vis=False, idxy=0):
        # generate mask
        if vis == True:
            bool_masked_pos = self._mask_center_test(center, idxy, noaug = noaug)
        else:
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)

        batch_size, seq_len, C = group_input_tokens.size()

        # add pos embedding
        # mask pos center
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C) 
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)

        x_vis = self.pointtransformer(x_vis, masked_center)

        pos = self.pos_embed(masked_center)
        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = utils.fps(xyz, self.num_group) 
        # knn to get the neighborhood
        idx = utils.knn(xyz, center, self.group_size) 
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head1 = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        y = self.head1(self.norm(x[:, -return_token_num:]))  # only return the visible patches
        return y


class Network(nn.Module):
    def __init__(self, config, mask_ratio):
        super(Network, self).__init__()
        super().__init__()
        self.config = config
        self.trans_dim = config['model']['trans_dim']
        self.MAE_encoder = MaskTransformer(config, mask_ratio)
        self.group_size = config['model']['group_size']
        self.num_group = config['model']['num_group']
        self.drop_path_rate = config['model']['drop_path_rate']
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config['model']['decoder_depth']
        self.decoder_num_heads = config['model']['decoder_num_heads']
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.MAE_decoder1 = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.MAE_decoder2 = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )  

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        self.query_embed = nn.Embedding(
            num_embeddings=config['model']['group_size'],
            embedding_dim=config['model']['trans_dim']
        ) 

        trunc_normal_(self.mask_token, std=.02)

    def forward(self, pts, idxy=0, vis=False):
        # Local patch masking
        neighborhood, center = self.group_divider(pts) 

        # Visible feature extraction
        x_vis, mask = self.MAE_encoder(neighborhood, center, vis=vis, idxy=idxy)  

        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)   
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)   
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)   

        # Iterative deformation
        _, M, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, M, -1)          

        x_full = torch.cat([x_vis, mask_token], dim=1)           
        mask_token_mid = self.MAE_decoder1(x_full, pos_full, M)
        masked_points_mid1 = self.increase_dim(mask_token_mid.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  

        x_full = torch.cat([x_vis, mask_token_mid], dim=1)
        mask_token_mid = self.MAE_decoder2(x_full, pos_full, M)
        masked_points_mid2 = self.increase_dim(mask_token_mid.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  

        x_full = torch.cat([x_vis, mask_token_mid], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, M)
        masked_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3) 
        gt_masked_points = neighborhood[mask].reshape(B * M, -1, 3) 

        if vis == True:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3) 
            full_vis = vis_points + center[~mask].unsqueeze(1)          
            full_rebuild = masked_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)               
            ret1 = full.reshape(-1, 3).unsqueeze(0)                      
            return ret1    
        else:
            return masked_points, masked_points_mid1, masked_points_mid2, gt_masked_points


class Network_Model(nn.Module):
    def __init__(self, config):
        super(Network_Model, self).__init__()
        super().__init__()

        self.config = config
        self.model_1 = Network(config, config['model']['mr6']).cuda()
        
    def forward(self, data, device, idxy=0, vis=False):
        if vis == True:
            pts_M1 = self.model_1(data, idxy=idxy, vis=True)
            pts_agr = pts_M1.reshape(1, -1, 3)
            
            return pts_agr
        
        else:
            masked_points, masked_points_mid1, masked_points_mid2, gpts_M1 = self.model_1(data)

            # Self-supervised completion
            loss_M = utils.get_emd_loss(masked_points, gpts_M1).unsqueeze(0)
            loss_M_mid1 = utils.get_emd_loss(masked_points_mid1, gpts_M1).unsqueeze(0)
            loss_M_mid2 = utils.get_emd_loss(masked_points_mid2, gpts_M1).unsqueeze(0)
            loss = loss_M + loss_M_mid1 + loss_M_mid2

            return loss
