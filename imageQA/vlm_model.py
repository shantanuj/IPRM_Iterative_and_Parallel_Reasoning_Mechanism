import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from einops import rearrange
from torch.nn.init import trunc_normal_
from basic_model_blocks import get_activation_layer, linear, length_to_mask, Transformer_Self_Attention_Block
from ipdb import set_trace as st
class Spatial_Position_Embedding(nn.Module):
    def __init__(self, 
                 num_pos_feats=512,
                 res_h = 14,
                 res_w = 14,
                 init_method = 'uniform',
                 dropout=0.1
                ):
        super().__init__()
        assert(num_pos_feats%2==0)
        self.row_embed = nn.Embedding(res_h, num_pos_feats//2)
        self.col_embed = nn.Embedding(res_w, num_pos_feats//2)
        self.pos_drop = nn.Dropout(dropout)
            
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        if(init_method in ['uniform']):
            nn.init.uniform_(self.row_embed.weight)
            nn.init.uniform_(self.col_embed.weight)
        elif(init_method in ['xavier_uniform']):
            nn.init.xavier_uniform_(self.row_embed.weight)
            nn.init.xavier_uniform_(self.col_embed.weight)
        elif(init_method in ['trunc_norm']):
            nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.col_embed.weight, std=0.02)
            
            
    def forward(self, x): #x is of shape B x C x H x W
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return self.pos_drop(pos + x)
    

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

class VLM_for_CLEVR(nn.Module):
    def __init__(
        self,
        cfg,
        ):
        super().__init__()
        
        self.cfg = cfg
        
        self.module_type = cfg.VLM.MODULE_TYPE
        
        dim = cfg.VLM.DIM
        classes = cfg.VLM.OUTPUT_CLASSES
        
        img_enc_dropout = cfg.VLM.VIS_STEM_DROPOUT
        lang_summary_rep_dropout = cfg.VLM.LANG_SUMMARY_REP_DROPOUT
        classifier_dropout = cfg.VLM.CLASSIFIER_DROPOUT
        
        self.conv = nn.Sequential(
            nn.Dropout(img_enc_dropout),
            nn.Conv2d(cfg.VLM.VIS_INPUT_FEATS, dim, 3, padding=1),
            get_activation_layer(cfg.VLM.VIS_STEM_NONLIN), 
            nn.Dropout(img_enc_dropout),
            nn.Conv2d(dim, dim, 3, padding=1),
            get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
        )
        self.add_positional_embds = cfg.VLM.ADD_VIS_POSITIONAL_EMBDS
        pos_drop_rate = cfg.VLM.POS_DROPOUT
        if(self.add_positional_embds):
            self.pos_layer = Spatial_Position_Embedding(num_pos_feats=dim,
                                                         res_h = cfg.VLM.INPUT_VIS_FEAT_RES[0],
                                                         res_w = cfg.VLM.INPUT_VIS_FEAT_RES[1],
                                                         init_method = 'uniform',
                                                         dropout = cfg.VLM.POS_DROPOUT
                                                        )
        

        if(self.cfg.VLM.USE_PRETRAINED_LANG_ENCODER):
            print("loading pretrained lang encoder from: ", cfg.VLM.PRETRAINED_LANG_ENCODER_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
            #if getattr(self.tokenizer, "pad_token", None) is None:
             #   self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Using AutoModel directly which loads RobertaModel (base) with pooler enabled by default
            self.lang_encoder = AutoModel.from_pretrained("distilbert/distilroberta-base")
            #st()
            
            lang_enc_dim = self.lang_encoder.embeddings.word_embeddings.weight.size(-1)
            self.proj_lang_summary = linear(lang_enc_dim, dim)
            self.proj_lang_tokens = linear(lang_enc_dim, dim)
        else:
            print("LOADING pretrained embeddings from: ", cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            self.tokenizer = torch.load(cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH, weights_only=False)
            self.embed = torch.load(cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH, weights_only=False)
            self.embed_hidden = embed_hidden = self.embed.word_embeddings.weight.size(-1)

            self.lstm = nn.LSTM(
                embed_hidden, dim // 2, batch_first=True, bidirectional=True
            )
        self.lang_summary_rep_pre_classifier_proj = linear(dim, dim)
        self.lang_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
        
        if(self.module_type in ['iprm']):
            from iprm_module import Recurrent_Memory_Attention_Module
            self.vlm_module = Recurrent_Memory_Attention_Module(dim, cfg)
        
        else:
            raise ValueError(f"Unsupported vlm module: {self.module_type}")
        
        if(cfg.VLM.CLASSIFIER_TYPE in ['only_result']):
            self.classifier = nn.Sequential(
                nn.Dropout(classifier_dropout),
                linear(dim, dim),
                get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN),
                nn.Dropout(classifier_dropout),
                linear(dim, classes),
            )
        elif(cfg.VLM.CLASSIFIER_TYPE in ['concat_lang_and_result']):
            self.classifier = nn.Sequential(
                nn.Dropout(classifier_dropout),
                linear(dim * 2, dim),
                get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN), 
                nn.Dropout(classifier_dropout),
                linear(dim, classes),
            )
        else:
            raise ValueError(f"Unsupported classifier module: {cfg.VLM.CLASSIFIER_TYPE}")
            
        
        #Initialize weights
        nn.init.xavier_uniform_(self.conv[1].weight)
        nn.init.zeros_(self.conv[1].bias)
        nn.init.xavier_uniform_(self.conv[4].weight)
        nn.init.zeros_(self.conv[4].bias)

        if(cfg.VLM.RESET_EMBED_LAYER):
            nn.init.uniform_(self.embed.word_embeddings.weight, -1.0, 1.0)
            
        self.add_aux_lang_token = cfg.VLM.ADD_AUX_LANG_TOKEN
        if(self.add_aux_lang_token):
            self.aux_lang_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.apply_aux_sa_layer = cfg.VLM.AUX_LANG_SA_LAYER
        if(self.apply_aux_sa_layer):
            self.lang_sa_block = Transformer_Self_Attention_Block(dim, 
                                                             num_heads=1,
                                                             mlp_inter_dim=4*dim,
                                                             dropout_att=0.0,
                                                             dropout_proj=cfg.VLM.LANG_ENC_DROPOUT,
                                                             mlp_act_layer=cfg.VLM.LANG_SA_ACT_LAYER
                                                            )
            
    def forward(self, vis_x, lang_x, save_atts=False):
        b = vis_x.size(0)
        if(self.cfg.VLM.IS_TRANSFORMER_VIS_FEAT): #vis_x is of shape BxNvxD and remove first class token
            vis_x = vis_x[:,1:,:]
            vis_x = vis_x.view(b, self.cfg.VLM.INPUT_VIS_FEAT_RES[0], self.cfg.VLM.INPUT_VIS_FEAT_RES[1], vis_x.size(-1))
            vis_x = vis_x.permute(0,3,1,2)
            
        vis_feats = self.conv(vis_x)
        if(self.add_positional_embds):
            vis_feats = self.pos_layer(vis_feats)
        vis_feats = rearrange(vis_feats, "b c h w -> b c (h w)")
        vis_feats_len, vis_feats_len_max = (
            torch.tensor([vis_feats.shape[2] for _ in range(vis_feats.shape[0])]),
            vis_feats.shape[2],
        )

        
            
        
        
        if(self.cfg.VLM.USE_PRETRAINED_LANG_ENCODER):
            tokenized_out = self.tokenizer(lang_x, return_tensors="pt", add_special_tokens=True, return_length=True, padding=True).to(
                vis_feats.device
            )
            lang_encoder_out = self.lang_encoder(
                    tokenized_out.input_ids,
                    attention_mask=tokenized_out.attention_mask,
                    return_dict=True,
                )
            #st()
            lang_tokens_rep = self.proj_lang_tokens(lang_encoder_out['last_hidden_state'])
            lang_summary_rep = self.lang_summary_rep_drop(self.proj_lang_summary(lang_encoder_out['pooler_output']))
            #lang_summary_rep = self.proj_lang_summary(lang_summary_rep)
            lang_tokens_len = tokenized_out['length'].to(vis_feats.device)
            if(self.cfg.VLM.REMOVE_FIRST_AND_LAST_SPECIAL_TOKENS_PRETRAINED_LANG_ENCODER):
                lang_tokens_rep = lang_tokens_rep[:,1:-1,:]#remove first and last token
                lang_tokens_len = lang_tokens_len - 2 #as we remove first and last tokens
            
        else:
            tokenized_out = self.tokenizer(lang_x, return_tensors="pt", add_special_tokens=False, return_length=True, padding=True).to(
            vis_feats.device
            )
            lang_tokens_rep = tokenized_out['input_ids'].to(vis_feats.device)
            lang_tokens_len = tokenized_out['length'].to(vis_feats.device)

            embed = self.embed(lang_tokens_rep)
        
            packed_embed = nn.utils.rnn.pack_padded_sequence(
                embed, lang_tokens_len.to('cpu'), batch_first=True, enforce_sorted = False
            )
        
        
            lstm_out_pack, (qvec_pack, _) = self.lstm(packed_embed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_pack, batch_first=True, total_length=lang_tokens_len.max()
            )
            lang_summary_rep = rearrange(qvec_pack, "dir b dim -> b (dir dim)") #dir represents 2 for bilstm
            lang_summary_rep = self.lang_summary_rep_drop(lang_summary_rep)
            lang_tokens_rep = lstm_out

        if(self.add_aux_lang_token):
            b = lang_tokens_rep.size(0)
            lang_tokens_rep = torch.cat([lang_tokens_rep, self.aux_lang_token.repeat(b, 1, 1)], dim=1)
            lang_tokens_len = lang_tokens_len + 1
        if(self.apply_aux_sa_layer):
            lang_token_mask = length_to_mask(lang_tokens_len, max_len=lang_tokens_rep.size(1)).to(lang_tokens_rep)
            lang_tokens_rep, _ = self.lang_sa_block(q_in = lang_tokens_rep, 
                                                  k_in = lang_tokens_rep, 
                                                  v_in = lang_tokens_rep, 
                                                  k_token_mask=lang_token_mask) 
            
        
        module_output = self.vlm_module(lang_tokens_rep, #B x Nl x D
                                        lang_summary_rep, #B x D
                                        vis_feats, #Bx D x Nv
                                        lang_tokens_len,
                                        vis_tokens_lens=vis_feats_len,
                                        vis_tokens_lens_max=vis_feats_len_max,
                                        save_atts =save_atts
                                       )
        
        if(self.cfg.VLM.CLASSIFIER_TYPE in ['only_result']):
            out = self.classifier(module_output)
        else:
            lang_summary_rep = self.lang_summary_rep_pre_classifier_proj(lang_summary_rep)
            out = torch.cat([module_output, lang_summary_rep], dim=1)
            out = self.classifier(out)

                
        return out
