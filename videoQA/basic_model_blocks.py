import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from einops import rearrange
from torchdiffeq import odeint  # adjoint took more nfe_backward and nfe_forward
from torch.nn.init import trunc_normal_, constant_

def get_activation_layer(activation_name):
    """
    Returns a PyTorch activation layer based on the input activation_name.

    Args:
        activation_name (str): Name of the activation function ('gelu', 'elu', 'relu', 'tanh', or 'sigmoid').

    Returns:
        nn.Module: PyTorch activation layer.
    """
    if activation_name.lower() == 'gelu':
        return nn.GELU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
        
def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


def linear2(in_dim, out_dim, bias=True, nonlin='elu'):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin2 = nn.Linear(out_dim, out_dim, bias=bias)
    nonlin = get_activation_layer(nonlin)
    xavier_uniform_(lin.weight)
    xavier_uniform_(lin2.weight)
    if bias:
        lin.bias.data.zero_()
        lin2.bias.data.zero_()

    return nn.Sequential(lin, nonlin, lin2)

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, "Length shape should be 1 dimensional."
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class Multi_Head_Attention_Block(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 dropout_att=0.1,
                 dropout_proj=0.1,
                 
                ):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout_att, batch_first=True)
        self.dropout_proj = nn.Dropout(dropout_proj)
        
    def forward(self, q_in, k_in, v_in, k_token_mask=None):
        if(k_token_mask is not None):
            mask = k_token_mask > 0 #k_token_mask is assumed to be an integer/bool mask (with 1s 1 for keep and 0s for not keep)
            mask = ~mask
        else:
            mask = k_token_mask
            
        mhsa_out, attn_wt = self.mhsa(q_in, k_in, v_in, key_padding_mask = mask)
        return self.dropout_proj(mhsa_out), attn_wt
    

class Multi_Head_Attention_Block_Two_Values(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 dropout_att=0.1,
                 dropout_proj=0.1,
                 
                ):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout_att, batch_first=True)
        #assert(num_heads==)
        if(num_heads!=1):
            raise ValueError("Error: Currently num_heads supported for 2 values input is only 1")
        self.dropout_proj = nn.Dropout(dropout_proj)
        self.v2_proj = linear(dim, dim)
        self.out2_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(dim, dim, bias=True)
        constant_(self.out2_proj.bias, 0.)
        self.drop2 = nn.Dropout(dropout_proj)
        
        
        
    def forward(self, q_in, k_in, v_in, v2_in, k_token_mask=None, attn_mask=None):
        if(k_token_mask is not None):
            mask = k_token_mask > 0 #k_token_mask is assumed to be an integer/bool mask (with 1s 1 for keep and 0s for not keep)
            mask = ~mask
        else:
            mask = k_token_mask
        
        
        mhsa_out, attn_wt = self.mhsa(q_in, k_in, v_in, key_padding_mask = mask, attn_mask = attn_mask)
        
        v2 = self.v2_proj(v2_in)
        out2 = torch.bmm(attn_wt, v2)
        out2 = self.out2_proj(out2)
        
        
        return self.dropout_proj(mhsa_out), self.drop2(out2), attn_wt
    
class Transformer_Self_Attention_Block(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 mlp_inter_dim = None,
                 dropout_att=0.1,
                 dropout_proj=0.1,
                 dropout_mlp=0.1,
                 mlp_act_layer='gelu',
                 ln_eps = 1e-5,
                 weighted_first_residual=False,
                 weighted_second_residual=False,
                 use_mlp=True
                ):
        super().__init__()
        self.mhsa_block = Multi_Head_Attention_Block(dim, num_heads, dropout_att, dropout_proj)
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.res1_wt = nn.Linear(dim, dim) if weighted_first_residual else nn.Identity()
        self.use_mlp = use_mlp
        if(use_mlp):
            if(mlp_inter_dim is None):
                mlp_inter_dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_inter_dim), 
                                     get_activation_layer(mlp_act_layer),
                                     nn.Dropout(dropout_mlp), 
                                     nn.Linear(mlp_inter_dim, dim),
                                     nn.Dropout(dropout_mlp)
                                    )
            self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
            self.res2_wt = nn.Linear(dim, dim) if weighted_second_residual else nn.Identity()
        
        
    def forward(self, 
                q_in, 
                k_in, 
                v_in, 
                apply_first_residual=True,
                apply_second_residual=True,
                k_token_mask=None,
                return_att_wts=True,
               ):
        out, attn_wt = self.mhsa_block(q_in, k_in, v_in, k_token_mask)
        if(apply_first_residual):
            out = self.norm1(self.res1_wt(q_in) + out)
        else:
            out = self.norm1(out)
            
        if(self.use_mlp):
            if(apply_second_residual):
                out = self.norm2(self.res2_wt(out) + self.mlp(out))
            else:
                out = self.norm2(self.mlp(out))
        if(return_att_wts):
            return out, attn_wt
        
        return out
        
        
class Transformer_Self_and_Cross_Attention_Block(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 mlp_inter_dim = None,
                 dropout_att=0.0,
                 dropout_proj=0.1,
                 dropout_mlp=0.1,
                 mlp_act_layer='gelu',
                 ln_eps = 1e-5,
                 weighted_first_residual=False,
                 weighted_second_residual=False,
                 weighted_third_residual=False,
                 use_mlp=True
                ):
        super().__init__()
        self.mhsa_block = Multi_Head_Attention_Block(dim, num_heads, dropout_att, dropout_proj)
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.res1_wt = nn.Linear(dim, dim) if weighted_first_residual else nn.Identity()
        
        self.mhca_block = Multi_Head_Attention_Block(dim, num_heads, dropout_att, dropout_proj)
        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.res2_wt = nn.Linear(dim, dim) if weighted_second_residual else nn.Identity()
        
        
        self.use_mlp = use_mlp
        if(use_mlp):
            if(mlp_inter_dim is None):
                mlp_inter_dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_inter_dim), 
                                     get_activation_layer(mlp_act_layer),
                                     nn.Dropout(dropout_mlp), 
                                     nn.Linear(mlp_inter_dim, dim),
                                     nn.Dropout(dropout_mlp)
                                    )
            self.norm3 = nn.LayerNorm(dim, eps=ln_eps)
            self.res3_wt = nn.Linear(dim, dim) if weighted_third_residual else nn.Identity()
        
        
    def forward(self, 
                q_in, 
                k_in, 
                v_in, 
                apply_first_residual=True,
                apply_second_residual=True,
                apply_third_residual=True,
                q_token_mask=None,
                k_token_mask=None,
                return_att_wts=True,
               ):
        out, sa_attn_wt = self.mhsa_block(q_in, q_in, q_in, q_token_mask)
        if(apply_first_residual):
            out = self.norm1(self.res1_wt(q_in) + out)
        else:
            out = self.norm1(out)
            
        out2, ca_attn_wt = self.mhca_block(out, k_in, v_in, k_token_mask)
        
        if(apply_second_residual):
            out = self.norm2(self.res2_wt(out) + out2)
        else:
            out = self.norm2(out2)
            
            
        if(self.use_mlp):
            if(apply_third_residual):
                out = self.norm3(self.res3_wt(out) + self.mlp(out))
            else:
                out = self.norm3(self.mlp(out))
        
        if(return_att_wts):
            return out, sa_attn_wt, ca_attn_wt
        
        return out


