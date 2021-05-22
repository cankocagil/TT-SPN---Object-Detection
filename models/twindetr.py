import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from siren.init import siren_uniform_





class TwinDetr(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):

        super(TwinDetr, self).__init__()

        # create ResNet-50 backbone
        self.backbone = TwinsSVT(
            num_classes = 1000,       # number of output classes
            s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
            s1_patch_size = 4,        # stage 1 - patch size for patch embedding
            s1_local_patch_size = 7,  # stage 1 - patch size for local attention
            s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
            s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
            s2_emb_dim = 128,         # stage 2 (same as above)
            s2_patch_size = 2,
            s2_local_patch_size = 7,
            s2_global_k = 7,
            s2_depth = 1,
            s3_emb_dim = 256,         # stage 3 (same as above)
            s3_patch_size = 2,
            s3_local_patch_size = 7,
            s3_global_k = 7,
            s3_depth = 5,
            s4_emb_dim = 512,         # stage 4 (same as above)
            s4_patch_size = 2,
            s4_local_patch_size = 7,
            s4_global_k = 7,
            s4_depth = 4,
            peg_kernel_size = 3,      # positional encoding generator kernel size
            dropout = 0.              # dropout
    )

        # create conversion layer
        self.conv = nn.Conv2d(512, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
            activation='gelu')

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = MLP(hidden_dim, 128, num_classes + 1) #nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = MLP(hidden_dim, 128, 4) #nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.sine_init(self.row_embed)
        self.sine_init(self.col_embed)

        self.sine = Sine(w0=1)


    def sine_init(self, x):
        siren_uniform_(x, mode='fan_in', c=6)
        

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)     

        # construct positional encodings
        H, W = h.shape[-2:]
        
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

    
        queries = self.query_pos.unsqueeze(1)


        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             queries).transpose(0, 1)
    
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}







class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)



class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dropout = 0.):
        super().__init__()
        internal_state_dim = int(hidden_dim//2)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, internal_state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(internal_state_dim, num_heads),
        )
    def forward(self, x):
        return self.net(x)
        
# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PatchEmbedding(nn.Module):
    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.proj = nn.Conv2d(patch_size ** 2 * dim, dim_out, 1)

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = p, p2 = p)
        return self.proj(fmap)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1))

    def forward(self, x):
        return self.proj(x)

class LocalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = - 1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h = h, x = x, y = y, p1 = p, p2 = p)
        return self.to_out(out)

class GlobalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., k = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride = k, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, dim_head = 64, mlp_mult = 4, local_patch_size = 7, global_k = 7, dropout = 0., has_local = True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LocalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_size = local_patch_size))) if has_local else nn.Identity(),
                Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))) if has_local else nn.Identity(),
                Residual(PreNorm(dim, GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, k = global_k))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout)))
            ]))
    def forward(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x

class TwinsSVT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_patch_size = 4,
        s1_local_patch_size = 7,
        s1_global_k = 7,
        s1_depth = 1,
        s2_emb_dim = 128,
        s2_patch_size = 2,
        s2_local_patch_size = 7,
        s2_global_k = 7,
        s2_depth = 1,
        s3_emb_dim = 256,
        s3_patch_size = 2,
        s3_local_patch_size = 7,
        s3_global_k = 7,
        s3_depth = 5,
        s4_emb_dim = 512,
        s4_patch_size = 2,
        s4_local_patch_size = 7,
        s4_global_k = 7,
        s4_depth = 4,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 3
        layers = []

        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'

            dim_next = config['emb_dim']

            layers.append(nn.Sequential(
                PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']),
                Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last),
                PEG(dim = dim_next, kernel_size = peg_kernel_size),
                Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)
            ))

            dim = dim_next

        self.layers = nn.Sequential(
            *layers,
            #nn.AdaptiveAvgPool2d(1),
            #Rearrange('... () () -> ...'),
            #nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)