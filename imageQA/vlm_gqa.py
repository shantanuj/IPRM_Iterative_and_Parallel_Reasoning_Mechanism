import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from einops import rearrange
from torch.nn.init import trunc_normal_
from basic_model_blocks import get_activation_layer, linear, length_to_mask, Transformer_Self_Attention_Block

class VLM_for_GQA(nn.Module):
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
        
        self.vis_proj1 = nn.Sequential(nn.Dropout(img_enc_dropout),
                                       nn.Conv1d(cfg.VLM.VIS_INPUT_FEATS, dim, 1, padding=0),
                                       get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                      )
        self.integrate_bbox_info = cfg.VLM.INTEGRATE_BBOX_INFO
        self.normalize_bbox = cfg.VLM.NORMALIZE_BBOX
        self.max_obj_proposals = cfg.VLM.MAX_OBJ_PROPOSALS
        
        #self.aux_layer = linear(4096, 10000)
        
        use_aux_proj = False
        aux_proj_dims = dim
        if(self.integrate_bbox_info):
            bbox_dim = cfg.VLM.BBOX_LATENT_DIMS
            use_aux_proj = True
            aux_proj_dims += bbox_dim
            
            self.bbox_proj = nn.Sequential(linear(4, bbox_dim), get_activation_layer(cfg.VLM.VIS_STEM_NONLIN))
            
            
        self.integrate_obj_label_feats = cfg.VLM.INTEGRATE_OBJ_LABEL_FEATS
        if(self.integrate_obj_label_feats):
            use_aux_proj = True
            self.obj_embed = torch.load(cfg.VLM.OBJ_LABEL_EMBD_LAYER_PATH)
            self.ln_obj_embed = nn.LayerNorm(300)
            self.dropout_obj_embed = nn.Dropout(0.1)
            obj_label_dim = cfg.VLM.OBJ_LABEL_LATENT_DIMS
            self.obj_label_proj = nn.Sequential(linear(300, obj_label_dim), get_activation_layer(cfg.VLM.VIS_STEM_NONLIN))
            aux_proj_dims += obj_label_dim

            
            
        if(use_aux_proj):
            self.aux_obj_bbox_vis_concat_proj = nn.Sequential(nn.Dropout(cfg.VLM.BBOX_VIS_CONCAT_DROPOUT),
                                                      linear(aux_proj_dims, dim), 
                                                      get_activation_layer(cfg.VLM.VIS_STEM_NONLIN))
            
      
        
        if(self.cfg.VLM.USE_PRETRAINED_LANG_ENCODER):
            print("loading pretrained lang encoder from: ", cfg.VLM.PRETRAINED_LANG_ENCODER_PATH)
            self.tokenizer = torch.load(cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH)
            self.lang_encoder = torch.load(cfg.VLM.PRETRAINED_LANG_ENCODER_PATH)
            lang_enc_dim = self.lang_encoder.embeddings.word_embeddings.weight.size(-1)
            self.proj_lang_summary = linear(lang_enc_dim, dim)
            self.proj_lang_tokens = linear(lang_enc_dim, dim)
        else:
            print("LOADING pretrained embeddings from: ", cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            self.tokenizer = torch.load(cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH)
            self.embed = torch.load(cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            self.embed_hidden = embed_hidden = self.embed.word_embeddings.weight.size(-1)

            self.lstm = nn.LSTM(
                embed_hidden, dim // 2, batch_first=True, bidirectional=True
            )
        
        
        self.lang_summary_rep_pre_classifier_proj = linear(dim, dim)
        self.lang_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
        
        if(self.module_type in ['rmam']):
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
        nn.init.xavier_uniform_(self.vis_proj1[1].weight)
        nn.init.zeros_(self.vis_proj1[1].bias)
        
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
            
            
        
            
            
            
            
            
    def forward(self, vis_x, lang_x, bbox_in, img_w, img_h, obj_labels, save_atts=False):
        #vis_x is of shape BxNxD where N can be spatial positions or object proposals
        b = vis_x.size(0)
        vis_feats = rearrange(vis_x, "b n c -> b c n")
        
        if(self.cfg.VLM.INPUT_IS_SPATIAL_FEATS):
            vis_feats_len, vis_feats_len_max = (
                torch.tensor([vis_feats.shape[2] for _ in range(vis_feats.shape[0])]),
                vis_feats.shape[2],
            )
        else:
            vis_feats_len, vis_feats_len_max = (vis_x.sum(dim=-1) != 0).long().sum(dim=-1), self.max_obj_proposals
            #print(f"CHECKING If VIS feats len is ok. it is {vis_feats_len} and max {vis_feats_len_max}")
            #input()
            #print(obj_labels[0])
            
            #vis_feats_len2 =  
            if(self.cfg.VLM.NORMALIZE_VIS_FEAT_PROPOSALS):
                vis_feats = F.normalize(vis_feats, dim=-1)
            
        vis_feats = self.vis_proj1(vis_feats) #->BxDxN output
        aux_vis_feats = None
        if(self.integrate_bbox_info):
            #normalize bboxes
            if(self.normalize_bbox):
                #print("Normalizing bbox")
                bbox_in[:,:,0] = bbox_in[:,:,0]/img_w.unsqueeze(1)
                bbox_in[:,:,2] = bbox_in[:,:,2]/img_w.unsqueeze(1)
                bbox_in[:,:,1] = bbox_in[:,:,1]/img_h.unsqueeze(1)
                bbox_in[:,:,3] = bbox_in[:,:,3]/img_h.unsqueeze(1)
            
            bbox_rep = self.bbox_proj(bbox_in) #B x Nx Db output
            aux_vis_feats = torch.cat([vis_feats.transpose(-1,-2), bbox_rep], dim=-1)
            

        if(self.integrate_obj_label_feats):
            obj_label_rep = self.ln_obj_embed(self.obj_embed(obj_labels))
            obj_label_rep = self.dropout_obj_embed(obj_label_rep)
            obj_label_rep = self.obj_label_proj(obj_label_rep)
            if(aux_vis_feats is None):
                aux_vis_feats = torch.cat([vis_feats.transpose(-1,-2), obj_label_rep], dim=-1)
            else:
                aux_vis_feats = torch.cat([aux_vis_feats, obj_label_rep], dim=-1)
                
        if(aux_vis_feats is not None):
            vis_feats = self.aux_obj_bbox_vis_concat_proj(aux_vis_feats)
            vis_feats = vis_feats.transpose(-1,-2) #BxNxD -> BxDxN
            
            
            
            
            
            
            
        
        if(self.cfg.VLM.USE_PRETRAINED_LANG_ENCODER):
            tokenized_out = self.tokenizer(lang_x, return_tensors="pt", add_special_tokens=True, return_length=True, padding=True).to(
                vis_feats.device
            )
            lang_encoder_out = self.lang_encoder(
                    tokenized_out.input_ids,
                    attention_mask=tokenized_out.attention_mask,
                    return_dict=True,
                )
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
            
        #print(3, vis_feats_len)
        #input()
        #print(4, vis_feats_len_max)
        #input()
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