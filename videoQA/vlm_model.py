import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from einops import rearrange
from torchdiffeq import odeint  # adjoint took more nfe_backward and nfe_forward
from torch.nn.init import trunc_normal_
from basic_model_blocks import get_activation_layer, linear, length_to_mask, Transformer_Self_Attention_Block
import copy
import torch.serialization
from copy import deepcopy
from lavis.models import load_model_and_preprocess
import lavis
import transformers
from transformers import AutoModel, AutoTokenizer
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])

#"""
from lavis.models import load_model_and_preprocess
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])
torch.serialization.add_safe_globals([lavis.processors.blip_processors.BlipQuestionProcessor])
torch.serialization.add_safe_globals([lavis.models.med.BertEmbeddings])
torch.serialization.add_safe_globals([torch.nn.modules.normalization.LayerNorm])
torch.serialization.add_safe_globals([torch.nn.modules.dropout.Dropout])
torch.serialization.add_safe_globals([transformers.models.bert.configuration_bert.BertConfig])
torch.serialization.add_safe_globals([transformers.models.bert.tokenization_bert.BertTokenizer])
torch.serialization.add_safe_globals([transformers.tokenization_utils.Trie])
torch.serialization.add_safe_globals([transformers.models.bert.tokenization_bert.BasicTokenizer])
torch.serialization.add_safe_globals([transformers.models.bert.tokenization_bert.WordpieceTokenizer])
#"""



class TFBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        relu_logits = F.relu(logits)
        neg_abs_logits = -torch.abs(logits)

        term1 = relu_logits - logits * targets
        term2 = torch.log1p(torch.exp(neg_abs_logits))
        loss = term1 + term2
        loss = loss.sum(dim=-1).mean(dim=-1)
        return loss
    
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
    


class VLM_for_STAR(nn.Module):
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
        
        
        
        use_aux_proj = False
        
        ##Person bbox+coords projections
        self.person_bbox_proj = nn.Sequential(linear(4, cfg.VLM.PERSON_BBOX_LATENT_DIMS), 
                                              get_activation_layer(cfg.VLM.VIS_STEM_NONLIN), 
                                              nn.Dropout(cfg.VLM.PERSON_BBOX_LATENT_DROP))
        person_feat_dims = cfg.VLM.PERSON_BBOX_LATENT_DIMS
        if(cfg.VLM.USE_PERSON_EMBEDDING):
            self.person_embd = nn.Parameter(torch.randn(1,1,1, cfg.VLM.PERSON_EMBD_DIMS))#repeat as BxFx1xD
            self.ln_person_embd = nn.LayerNorm(cfg.VLM.PERSON_EMBD_DIMS)
            self.dropout_person_embd = nn.Dropout(cfg.VLM.PERSON_EMBD_DROPOUT)
            self.person_embd_proj = nn.Sequential(linear(cfg.VLM.PERSON_EMBD_DIMS, cfg.VLM.PERSON_EMBD_LATENT_DIMS), 
                                                get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                                nn.Dropout(cfg.VLM.PERSON_EMBED_LATENT_DROP)
                                               )
            person_feat_dims += cfg.VLM.PERSON_EMBD_LATENT_DIMS
            
        if(cfg.VLM.PERSON_COORDS_LATENT_METHOD in ['concat_coords_then_proj']):
            self.person_coords_proj = nn.Sequential(linear(2* cfg.VLM.NUM_PERSON_KEYPOINTS, cfg.VLM.PERSON_COORDS_LATENT_DIMS),
                                                    get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                                    nn.Dropout(cfg.VLM.PERSON_COORDS_LATENT_DROP)
                                                   )
            person_feat_dims += cfg.VLM.PERSON_COORDS_LATENT_DIMS
            
        elif(cfg.VLM.PERSON_COORDS_LATENT_METHOD in ['indv_projs_then_sum']):
            coords_projs_list = [nn.Sequential(linear(2, cfg.VLM.PERSON_COORDS_LATENT_DIMS), 
                                               get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                               nn.Dropout(cfg.VLM.PERSON_COORDS_LATENT_DROP)
                                              )
                                 for i in range(cfg.VLM.NUM_PERSON_KEYPOINTS)]
            self.person_coords_projs = nn.ModuleList(coords_projs_list)
            person_feat_dims += cfg.VLM.PERSON_COORDS_LATENT_DIMS
            
        elif(cfg.VLM.PERSON_COORDS_LATENT_METHOD in ['not_used']):
            None
        else:
            raise ValueError(f"unsupported VLM.PERSON_COORDS_LATENT_METHOD {VLM.PERSON_COORDS_LATENT_METHOD}")
        
        self.person_coords_and_bbox_concat_proj = nn.Sequential(nn.Dropout(cfg.VLM.CONCAT_PERSON_VIS_DROPOUT),
                                                                linear(person_feat_dims, dim), 
                                                                get_activation_layer(cfg.VLM.VIS_STEM_NONLIN))
        
        self.obj_bbox_proj = nn.Sequential(linear(4, cfg.VLM.OBJ_BBOX_LATENT_DIMS), 
                                           get_activation_layer(cfg.VLM.VIS_STEM_NONLIN), 
                                           nn.Dropout(cfg.VLM.OBJ_BBOX_LATENT_DROP))
        
        print(f"loading attrb embedding layer from: {cfg.VLM.OBJ_LABEL_EMBD_LAYER_PATH}")
        self.obj_label_embed = torch.load(cfg.VLM.OBJ_LABEL_EMBD_LAYER_PATH, weights_only=True)
        self.ln_obj_label_embed = nn.LayerNorm(cfg.VLM.OBJ_LABEL_EMBD_DIMS)
        self.dropout_obj_label_embed = nn.Dropout(cfg.VLM.OBJ_LABEL_EMBD_DROPOUT)
        self.obj_label_proj = nn.Sequential(linear(cfg.VLM.OBJ_LABEL_EMBD_DIMS, cfg.VLM.OBJ_LABEL_LATENT_DIMS), 
                                            get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                            nn.Dropout(cfg.VLM.OBJ_LABEL_LATENT_DROP)
                                           )
        
        if(cfg.VLM.USE_COMBINED_OBJ_REL_EMBED):
            self.obj_rels_wts = nn.Parameter(torch.randn(cfg.VLM.NUM_OBJ_RELS, cfg.VLM.OBJ_REL_EMBD_DIMS))
            self.ln_obj_rel_embed = nn.LayerNorm(cfg.VLM.OBJ_REL_EMBD_DIMS)
            self.dropout_obj_rel_embed = nn.Dropout(cfg.VLM.OBJ_REL_EMBD_DROPOUT)
            self.obj_rel_proj = nn.Sequential(linear(cfg.VLM.OBJ_REL_EMBD_DIMS, cfg.VLM.OBJ_REL_LATENT_DIMS), 
                                                get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                                nn.Dropout(cfg.VLM.OBJ_REL_LATENT_DROP)
                                               )
            obj_concat_vis_dims = cfg.VLM.OBJ_REL_LATENT_DIMS + cfg.VLM.OBJ_LABEL_LATENT_DIMS + cfg.VLM.OBJ_BBOX_LATENT_DIMS
        else:
            self.obj_att_rels_wts = nn.Parameter(torch.randn(cfg.VLM.NUM_OBJ_ATT_RELS, cfg.VLM.OBJ_REL_EMBD_DIMS))
            self.ln_obj_att_rel_embed = nn.LayerNorm(cfg.VLM.OBJ_REL_EMBD_DIMS)
            self.dropout_obj_att_rel_embed = nn.Dropout(cfg.VLM.OBJ_REL_EMBD_DROPOUT)
            self.obj_att_rel_proj = nn.Sequential(linear(cfg.VLM.OBJ_REL_EMBD_DIMS, cfg.VLM.OBJ_REL_LATENT_DIMS), 
                                                get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                                nn.Dropout(cfg.VLM.OBJ_REL_LATENT_DROP)
                                               )
            self.obj_spatial_rels_wts = nn.Parameter(torch.randn(cfg.VLM.NUM_OBJ_SPATIAL_RELS, cfg.VLM.OBJ_REL_EMBD_DIMS))
            self.ln_obj_spatial_rel_embed = nn.LayerNorm(cfg.VLM.OBJ_REL_EMBD_DIMS)
            self.dropout_obj_spatial_rel_embed = nn.Dropout(cfg.VLM.OBJ_REL_EMBD_DROPOUT)
            self.obj_spatial_rel_proj = nn.Sequential(linear(cfg.VLM.OBJ_REL_EMBD_DIMS, cfg.VLM.OBJ_REL_LATENT_DIMS), 
                                                get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                                nn.Dropout(cfg.VLM.OBJ_REL_LATENT_DROP)
                                               )
            self.obj_contact_rels_wts = nn.Parameter(torch.randn(cfg.VLM.NUM_OBJ_CONTACT_RELS, cfg.VLM.OBJ_REL_EMBD_DIMS))
            self.ln_obj_contact_rel_embed = nn.LayerNorm(cfg.VLM.OBJ_REL_EMBD_DIMS)
            self.dropout_obj_contact_rel_embed = nn.Dropout(cfg.VLM.OBJ_REL_EMBD_DROPOUT)
            self.obj_contact_rel_proj = nn.Sequential(linear(cfg.VLM.OBJ_REL_EMBD_DIMS, cfg.VLM.OBJ_REL_LATENT_DIMS), 
                                                get_activation_layer(cfg.VLM.VIS_STEM_NONLIN),
                                                nn.Dropout(cfg.VLM.OBJ_REL_LATENT_DROP)
                                               )
            obj_concat_vis_dims = 3*cfg.VLM.OBJ_REL_LATENT_DIMS + cfg.VLM.OBJ_LABEL_LATENT_DIMS + cfg.VLM.OBJ_BBOX_LATENT_DIMS
            
            
            
            
        self.obj_bbox_label_rel_concat_proj = nn.Sequential(nn.Dropout(cfg.VLM.CONCAT_OBJ_VIS_DROPOUT),
                                                            linear(obj_concat_vis_dims, dim), 
                                                              get_activation_layer(cfg.VLM.VIS_STEM_NONLIN))
        
        self.use_frame_positional_embed = cfg.VLM.USE_FRAME_POS_EMBEDS
        
        
        if(self.use_frame_positional_embed):
            self.frame_positional_embed = nn.Parameter(torch.randn(1, cfg.DATALOADER.NUM_SAMPLE_FRAMES, 1, dim)) #BxFxNxD
            
        
        ####Language module
        #self.aux_layer = linear(4096,150000)
        if(self.cfg.VLM.USE_PRETRAINED_LANG_ENCODER):
            print("loading pretrained lang encoder from: ", cfg.VLM.PRETRAINED_LANG_ENCODER_PATH)
            self.lang_encoder = AutoModel.from_pretrained("distilbert/distilroberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
            #self.lang_encoder = torch.load(cfg.VLM.PRETRAINED_LANG_ENCODER_PATH)    
            #self.tokenizer = torch.load(cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH)
            
            #del vqa_model, vqa2, txt

            lang_enc_dim = self.lang_encoder.embeddings.word_embeddings.weight.size(-1)
            self.proj_lang_summary = linear(lang_enc_dim, dim)
            self.proj_lang_tokens = linear(lang_enc_dim, dim)
            if(self.cfg.VLM.SHARE_QUESTION_AND_OPTION_ENCODER):
                self.option_encoder = self.lang_encoder
            else:
                self.option_encoder = copy.deepcopy(self.lang_encoder)
            if(self.cfg.VLM.SHARE_QUESTION_AND_OPTION_PROJS_METHOD in ['share_question_and_options']):
                print("==> Sharing projections amongst question + options")
                self.proj_option0_summary = self.proj_lang_summary
                self.proj_option0_tokens = self.proj_lang_tokens
                self.proj_option1_summary = self.proj_lang_summary
                self.proj_option1_tokens = self.proj_lang_tokens
                self.proj_option2_summary = self.proj_lang_summary
                self.proj_option2_tokens = self.proj_lang_tokens
                self.proj_option3_summary = self.proj_lang_summary
                self.proj_option3_tokens = self.proj_lang_tokens
                self.option0_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option1_summary_rep_drop = self.option0_summary_rep_drop
                self.option2_summary_rep_drop = self.option0_summary_rep_drop
                self.option3_summary_rep_drop = self.option0_summary_rep_drop
        
            elif(self.cfg.VLM.SHARE_QUESTION_AND_OPTION_PROJS_METHOD in ['share_options']):
                print("==> Sharing projections amongst options")
                self.proj_option0_summary = linear(lang_enc_dim, dim)
                self.proj_option0_tokens = linear(lang_enc_dim, dim)
                self.proj_option1_summary = self.proj_option0_summary
                self.proj_option1_tokens = self.proj_option0_tokens
                self.proj_option2_summary = self.proj_option0_summary
                self.proj_option2_tokens = self.proj_option0_tokens
                self.proj_option3_summary = self.proj_option0_summary
                self.proj_option3_tokens = self.proj_option0_tokens
                self.option0_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option1_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option2_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option3_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
            else:
                print("==> Unique projections for each question as well as option")
                
                self.proj_option0_summary = linear(lang_enc_dim, dim)
                self.proj_option0_tokens = linear(lang_enc_dim, dim)
                self.proj_option1_summary = linear(lang_enc_dim, dim)
                self.proj_option1_tokens = linear(lang_enc_dim, dim)
                self.proj_option2_summary = linear(lang_enc_dim, dim)
                self.proj_option2_tokens = linear(lang_enc_dim, dim)
                self.proj_option3_summary = linear(lang_enc_dim, dim)
                self.proj_option3_tokens = linear(lang_enc_dim, dim)
                self.option0_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option1_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option2_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
                self.option3_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
        
                
            
        else:
            #print("LOADING pretrained embeddings from: ", cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            #vqa_model, vqa2, txt = load_model_and_preprocess(name='blip_vqa', model_type='vqav2', is_eval=True, device='cpu')
            #self.tokenizer = txt['eval']
            #self.embed= deepcopy(vqa_model.text_encoder.embeddings)
            #del vqa_model, vqa2, txt
            print("loading pretrained lang embedding from: ", cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            self.embed = torch.load(cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            # print("loading pretrained lang tokenizer from: ", cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH)
            # self.tokenizer = torch.load(cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH)
            print("Loading fresh BertTokenizer (bert-base-uncased) to resolve version compatibility issues...")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
            #self.embed = torch.load(cfg.VLM.PRETRAINED_LANG_EMBEDDING_PATH)
            #self.embed = vqa_model.text_encoder.embeddings.word_embeddings
            self.embed_hidden = embed_hidden = self.embed.word_embeddings.weight.size(-1)
            self.option0_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
            self.option1_summary_rep_drop = self.option0_summary_rep_drop
            self.option2_summary_rep_drop = self.option0_summary_rep_drop
            self.option3_summary_rep_drop = self.option0_summary_rep_drop
            self.lstm = nn.LSTM(
                embed_hidden, dim // 2, batch_first=True, bidirectional=True
            )
            if(cfg.VLM.LSTM_APPLY_AUX_PROJS):
                self.proj_lang_summary = linear(dim, dim)
                self.proj_lang_tokens = linear(dim, dim)
                self.proj_option0_summary = linear(dim, dim)
                self.proj_option0_tokens = linear(dim, dim)
                self.proj_option1_summary = self.proj_option0_summary
                self.proj_option1_tokens = self.proj_option0_tokens
                self.proj_option2_summary = self.proj_option0_summary
                self.proj_option2_tokens = self.proj_option0_tokens
                self.proj_option3_summary = self.proj_option0_summary
                self.proj_option3_tokens = self.proj_option0_tokens
            else:
                self.proj_lang_summary = nn.Identity()
                self.proj_lang_tokens = nn.Identity()
                self.proj_option_summary = nn.Identity()
                self.proj_option_tokens = nn.Identity()
                self.proj_option0_summary = nn.Identity()
                self.proj_option0_tokens = nn.Identity()
                self.proj_option1_summary = nn.Identity()
                self.proj_option1_tokens = nn.Identity()
                self.proj_option2_summary = nn.Identity()
                self.proj_option2_tokens = nn.Identity()
                self.proj_option3_summary = nn.Identity()
                self.proj_option3_tokens = nn.Identity()
                
            if(self.cfg.VLM.SHARE_QUESTION_AND_OPTION_ENCODER):
                self.option_lstm = self.lstm
            else:
                self.option_lstm = nn.LSTM(
                embed_hidden, dim // 2, batch_first=True, bidirectional=True
            )
        
        self.lang_summary_rep_pre_classifier_proj = linear(dim, dim)
        self.lang_summary_rep_drop = nn.Dropout(lang_summary_rep_dropout)
        self.option_summary_rep_pre_classifier_proj = linear(dim, dim)
        if(self.cfg.VLM.COMBINE_QUES_AND_OPTION_SUMMARY_FOR_MODULE):
            self.combine_ques_and_option_summary_proj = linear(2*dim, dim)
            self.combine_ques_and_option_summary_drop = nn.Dropout(lang_summary_rep_dropout)
        if(cfg.VLM.ADD_LANG_AND_CHOICE_POSITIONAL_EMBEDS):
            self.question_lang_pos = nn.Parameter(torch.randn(1, 1, dim))
            self.option_lang_pos = nn.Parameter(torch.randn(1, 1, dim))
            
            
        if(self.module_type in ['iprm']):
            from iprm_module import Recurrent_Memory_Attention_Module
            self.vlm_module = Recurrent_Memory_Attention_Module(dim, cfg)
        elif(self.module_type in ['iprm_v2']):
            from iprm_v2_module import Recurrent_Memory_Attention_Module
            self.vlm_module = Recurrent_Memory_Attention_Module(dim, cfg)
        
        else:
            raise ValueError(f"Unsupported vlm module: {self.module_type}")
        
        if(cfg.VLM.PROCESS_EACH_OPTION_INDEPENDENTLY):
            if(cfg.VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE in ['joint_concat']):
                self.concat_classifier = nn.Sequential(#nn.Dropout(cfg.VLM.CONCAT_CLASSIFIER_DROP),
                                                           linear(4*cfg.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM, classes),
                                                           
                                                            #linear(4*cfg.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM, dim),
                                                            #get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN),
                                                            #nn.Dropout(classifier_dropout),
                                                            #linear(dim, classes)
                                                          )
            elif(cfg.VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE in ['individual']):
                self.indv_classifier = nn.Sequential(#nn.Dropout(cfg.VLM.CONCAT_CLASSIFIER_DROP),
                                                       linear(cfg.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM, 1)
                                                          )
            else:
                raise ValueError(f"Unsupported INDEPENDENT_OPTION_CLASSIFIER_TYPE: {cfg.VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE}")    
                
            if(cfg.VLM.CLASSIFIER_TYPE in ['only_result']):
                self.classifier = nn.Sequential(
                    nn.Dropout(classifier_dropout),
                    linear(dim, cfg.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM),
                    get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN),
                    nn.Dropout(cfg.VLM.INDV_OPTION_CLASSIFIER_DROP),
                    #linear(dim, cfg.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM),
                    #get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN),
                )
                
            elif(cfg.VLM.CLASSIFIER_TYPE in ['concat_lang_and_result']):
                self.classifier = nn.Sequential(
                    nn.Dropout(classifier_dropout),
                    linear(dim * 3, cfg.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM),
                    get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN), 
                    nn.Dropout(cfg.VLM.INDV_OPTION_CLASSIFIER_DROP),
                    #linear(dim, cfg.VLM.INDIVIDUAL_CLASSIFIER_OPTION_DIM),
                    #get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN),
                )
            else:
                raise ValueError(f"Unsupported classifier module: {cfg.VLM.CLASSIFIER_TYPE}")

                
        else:
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
                    linear(dim * 3, dim),
                    get_activation_layer(cfg.VLM.CLASSIFIER_NONLIN), 
                    nn.Dropout(classifier_dropout),
                    linear(dim, classes),
                )
            else:
                raise ValueError(f"Unsupported classifier module: {cfg.VLM.CLASSIFIER_TYPE}")

        if(cfg.VLM.RESET_OBJ_LABEL_EMBED_LAYER):
            print("WARNING: Resetting obj attribute embedding layer")
            nn.init.uniform_(self.obj_attrb_embed.weight, -1.0, 1.0)
        
        if(cfg.VLM.UNIFORM_INIT_OBJ_RELS_EMBED):
            print("WARNING: Resetting obj rel labels layers")
            if(cfg.VLM.USE_COMBINED_OBJ_REL_EMBED):
                nn.init.uniform_(self.obj_rels_wts, -1.0, 1.0)
            else:
                nn.init.uniform_(self.obj_att_rels_wts, -1.0, 1.0)
                nn.init.uniform_(self.obj_spatial_rels_wts, -1.0, 1.0)
                nn.init.uniform_(self.obj_contact_rels_wts, -1.0, 1.0)
        
        if(cfg.VLM.UNIFORM_INIT_PERSON_EMBD and cfg.VLM.USE_PERSON_EMBEDDING):
            print("WARNING: Uniform init person embd")
            nn.init.uniform_(self.person_embd, -1.0, 1.0)
            
        if(cfg.VLM.RESET_EMBED_LAYER):
            print("WARNING: Resetting word embedding layer")
            nn.init.uniform_(self.embed.word_embeddings.weight, -1.0, 1.0)
            
            
            
        self.add_aux_lang_token = cfg.VLM.ADD_AUX_LANG_TOKEN
        if(self.add_aux_lang_token):
            self.aux_lang_token = nn.Parameter(torch.randn(1, 1, dim))
        
        
            
            
            
            
    def _run_lang_encoder_transformer(self,
                                      in_x,
                                      lang_encoder_model,
                                      token_proj_layer,
                                      summary_proj_layer,
                                      summary_drop_layer,
                                      device
                                     ):
        tokenized_out = self.tokenizer(in_x, return_tensors="pt", add_special_tokens=True, return_length=True, padding=True).to(
                device
            )
        lang_encoder_out = lang_encoder_model(
                tokenized_out.input_ids,
                attention_mask=tokenized_out.attention_mask,
                return_dict=True,
            )
        lang_tokens_rep = token_proj_layer(lang_encoder_out['last_hidden_state'])
        lang_summary_rep = summary_drop_layer(summary_proj_layer(lang_encoder_out['pooler_output']))
        #lang_summary_rep = self.proj_lang_summary(lang_summary_rep)
        lang_tokens_len = tokenized_out['length'].to(device)
        if(self.cfg.VLM.REMOVE_FIRST_AND_LAST_SPECIAL_TOKENS_PRETRAINED_LANG_ENCODER):
            lang_tokens_rep = lang_tokens_rep[:,1:-1,:]#remove first and last token
            lang_tokens_len = lang_tokens_len - 2 #as we remove first and last tokens
        return lang_summary_rep, lang_tokens_rep, lang_tokens_len
    
    def _run_lang_encoder_lstm(self,
                               in_x,
                               embed_layer,
                               lstm_block,
                               lang_summary_drop_layer,
                               token_proj_layer,
                               summary_proj_layer,        
                               device
                              ):
        tokenized_out = self.tokenizer(in_x, return_tensors="pt", add_special_tokens=False, return_length=True, padding=True).to(device)
        lang_tokens_rep = tokenized_out['input_ids'].to(device)
        lang_tokens_len = tokenized_out['length'].to(device)

        embed = embed_layer(lang_tokens_rep)

        packed_embed = nn.utils.rnn.pack_padded_sequence(
            embed, lang_tokens_len.to('cpu'), batch_first=True, enforce_sorted = False
        )


        lstm_out_pack, (qvec_pack, _) = lstm_block(packed_embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_pack, batch_first=True, total_length=lang_tokens_len.max()
        )
        lang_summary_rep = rearrange(qvec_pack, "dir b dim -> b (dir dim)") #dir represents 2 for bilstm
        lang_summary_rep = lang_summary_drop_layer(summary_proj_layer(lang_summary_rep))
        lang_tokens_rep = token_proj_layer(lstm_out)
        return lang_summary_rep, lang_tokens_rep, lang_tokens_len

    def forward(self, 
                person_vis_x_tuple, #(person_bbox_x, person_keypoints_x)
                objs_vis_x_tuple, #(obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x), #this tuple = vis_x
                lang_x_tuple, #(lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x), #this tuple = lang_x
                objs_resolutions, #of shape BxFx2 (used to normalize bbox)
                num_objs, #of shape BxF (list used to make mask)
                num_unpadded_frames, #of shape B (list used to make mask)
                save_atts=False):
        if(self.cfg.VLM.PROCESS_EACH_OPTION_INDEPENDENTLY):
            return self.forward_run_each_option_separately(person_vis_x_tuple, #(person_bbox_x, person_keypoints_x)
                                                            objs_vis_x_tuple, #(obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x), #this tuple = vis_x
                                                            lang_x_tuple, #(lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x), #this tuple = lang_x
                                                            objs_resolutions, #of shape BxFx2 (used to normalize bbox)
                                                            num_objs, #of shape BxF (list used to make mask)
                                                            num_unpadded_frames, #of shape B (list used to make mask)
                                                            save_atts)
        else:
            return self.forward_run_options_together(person_vis_x_tuple, #(person_bbox_x, person_keypoints_x)
                                                    objs_vis_x_tuple, #(obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x), #this tuple = vis_x
                                                    lang_x_tuple, #(lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x), #this tuple = lang_x
                                                    objs_resolutions, #of shape BxFx2 (used to normalize bbox)
                                                    num_objs, #of shape BxF (list used to make mask)
                                                    num_unpadded_frames, #of shape B (list used to make mask)
                                                    save_atts)
                
                
    def forward_run_each_option_separately(self, 
                person_vis_x_tuple, #(person_bbox_x, person_keypoints_x)
                objs_vis_x_tuple, #(obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x), #this tuple = vis_x
                lang_x_tuple, #(lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x), #this tuple = lang_x
                objs_resolutions, #of shape BxFx2 (used to normalize bbox)
                num_objs, #of shape BxF (list used to make mask)
                num_unpadded_frames, #of shape B (list used to make mask)
                save_atts=False):
        person_bbox_x, person_keypoints_x = person_vis_x_tuple
        person_keypoints_x = person_keypoints_x.squeeze(2) #B x F x 1 x N_k x 3 -> B x F x N_k x 3
        person_keypoints_x = person_keypoints_x[:,:,:,:2] #B x F x N_k x 3 -> #B x F x N_k x 2
        obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x = objs_vis_x_tuple
        lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x = lang_x_tuple
        
        assert(obj_bbox_x.size(1)==self.cfg.DATALOADER.NUM_SAMPLE_FRAMES)
        
        b = obj_bbox_x.size(0)
        f = obj_bbox_x.size(1)
        if(self.cfg.VLM.NORMALIZE_BBOX_AND_PERSON_COORDS):
                #print("Normalizing bbox")
            vis_w = objs_resolutions[:,:,1].unsqueeze(-1) #BxFx2 -> BxFx1
            vis_h = objs_resolutions[:,:,0].unsqueeze(-1) #BxFx2 -> BxFx1
            person_bbox_x[:,:,:,0] = person_bbox_x[:,:,:,0]/vis_w #normalize width
            person_bbox_x[:,:,:,2] = person_bbox_x[:,:,:,2]/vis_w
            person_bbox_x[:,:,:,1] = person_bbox_x[:,:,:,1]/vis_h
            person_bbox_x[:,:,:,3] = person_bbox_x[:,:,:,3]/vis_h
            
            obj_bbox_x[:,:,:,0] = obj_bbox_x[:,:,:,0]/vis_w #normalize width
            obj_bbox_x[:,:,:,2] = obj_bbox_x[:,:,:,2]/vis_w
            obj_bbox_x[:,:,:,1] = obj_bbox_x[:,:,:,1]/vis_h
            obj_bbox_x[:,:,:,3] = obj_bbox_x[:,:,:,3]/vis_h
            
            person_keypoints_x[:,:,:,0] = person_keypoints_x[:,:,:,0]/vis_w#.unsqueeze( #normalize width
            person_keypoints_x[:,:,:,1] = person_keypoints_x[:,:,:,0]/vis_h#.unsqueeze( #normalize width
            
        person_keypoints_x = person_keypoints_x.unsqueeze(2) #-> BxFx1xNkx2
        person_bbox_latent_rep = self.person_bbox_proj(person_bbox_x) #->BxFx1xD
            
        if(self.cfg.VLM.PERSON_COORDS_LATENT_METHOD in ['concat_coords_then_proj']):
            person_keypoints_x = person_keypoints_x.flatten(start_dim=3) #->BxFx1x(Nkx2)
            person_keypoints_latent_rep = self.person_coords_proj(person_keypoints_x) #->BxFx1xD
            person_keypoint_bbox_concat_rep = torch.cat([person_bbox_latent_rep, person_keypoints_latent_rep], dim=-1) 
        elif(self.cfg.VLM.PERSON_COORDS_LATENT_METHOD in ['indv_projs_then_sum']):
            for ky_i in range(person_keypoints_x.size(3)):
                if(ky_i==0):
                    person_keypoints_latent_rep = self.person_coords_projs[ky_i](person_keypoints_x[:,:,:,ky_i,:])
                else:
                    person_keypoints_latent_rep = person_keypoints_latent_rep + self.person_coords_projs[ky_i](person_keypoints_x[:,:,:,ky_i,:])
            person_keypoint_bbox_concat_rep = torch.cat([person_bbox_latent_rep, person_keypoints_latent_rep], dim=-1) 
        else:
            person_keypoint_bbox_concat_rep = person_bbox_latent_rep
        
        
        if(self.cfg.VLM.USE_PERSON_EMBEDDING):
            person_label_latent_rep = self.person_embd_proj(self.dropout_person_embd(self.ln_person_embd(self.person_embd)))
            person_label_latent_rep = person_label_latent_rep.repeat(b, f, 1, 1)
            person_keypoint_bbox_concat_rep = torch.cat([person_keypoint_bbox_concat_rep, person_label_latent_rep], dim=-1) 
        
            
        person_keypoint_bbox_latent_rep = self.person_coords_and_bbox_concat_proj(person_keypoint_bbox_concat_rep) #->BxFx1xD
        
        
        obj_bbox_latent_rep = self.obj_bbox_proj(obj_bbox_x)
        
        obj_label_latent_rep = self.obj_label_proj(self.dropout_obj_label_embed(self.ln_obj_label_embed(self.obj_label_embed(obj_label_x))))
        
        if(self.cfg.VLM.USE_COMBINED_OBJ_REL_EMBED):
            obj_combined_rel_latent_rep = combined_rel_x.unsqueeze(-1) * self.obj_rels_wts.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            obj_combined_rel_latent_rep = self.obj_rel_proj(self.dropout_obj_rel_embed(self.ln_obj_rel_embed(obj_combined_rel_latent_rep)))
            obj_combined_rel_latent_rep = obj_combined_rel_latent_rep.sum(dim=-2)#BxFxNxNrxD -> BxFxNxD
            obj_bbox_label_rels_concat_rep = torch.cat([obj_bbox_latent_rep, obj_label_latent_rep, obj_combined_rel_latent_rep], dim=-1)
        
        else:
            obj_att_rel_latent_rep = att_rel_x.unsqueeze(-1) * self.obj_att_rels_wts.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            obj_att_rel_latent_rep = self.obj_att_rel_proj(self.dropout_obj_att_rel_embed(self.ln_obj_att_rel_embed(obj_att_rel_latent_rep)))
            obj_att_rel_latent_rep = obj_att_rel_latent_rep.sum(dim=-2) #BxFxNxNrxD -> BxFxNxD
            
            obj_spatial_rel_latent_rep = spatial_rel_x.unsqueeze(-1) * self.obj_spatial_rels_wts.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            obj_spatial_rel_latent_rep = self.obj_spatial_rel_proj(self.dropout_obj_spatial_rel_embed(self.ln_obj_spatial_rel_embed(obj_spatial_rel_latent_rep)))
            obj_spatial_rel_latent_rep = obj_spatial_rel_latent_rep.sum(dim=-2)#BxFxNxNrxD -> BxFxNxD
            
            obj_contact_rel_latent_rep = contact_rel_x.unsqueeze(-1) * self.obj_contact_rels_wts.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            obj_contact_rel_latent_rep = self.obj_contact_rel_proj(self.dropout_obj_contact_rel_embed(self.ln_obj_contact_rel_embed(obj_contact_rel_latent_rep)))
            obj_contact_rel_latent_rep = obj_contact_rel_latent_rep.sum(dim=-2)#BxFxNxNrxD -> BxFxNxD
            
            obj_bbox_label_rels_concat_rep = torch.cat([obj_bbox_latent_rep, obj_label_latent_rep, obj_att_rel_latent_rep, obj_spatial_rel_latent_rep, obj_contact_rel_latent_rep], dim=-1)
            
        
        obj_bbox_label_rels_concat_rep = self.obj_bbox_label_rel_concat_proj(obj_bbox_label_rels_concat_rep) #->BxFxNoxD
        vis_feats = torch.cat([person_keypoint_bbox_latent_rep, obj_bbox_label_rels_concat_rep], dim=2) #->BxFx(No+1)xD
        
        if(self.use_frame_positional_embed):
            vis_feats = vis_feats + self.frame_positional_embed #BxFxNxD
        
        vis_framewise_masks = []
        num_objs_tensor = torch.tensor(num_objs) + 1 #add 1 for person token
                                       
        max_objs_len = vis_feats.size(2)
        for b_i in range(vis_feats.size(0)):
            num_objs_tensor[b_i][num_unpadded_frames[b_i]:] = 0 
        for f in range(vis_feats.size(1)):
            vis_framewise_masks.append(length_to_mask(num_objs_tensor[:,f], max_len=max_objs_len).to(vis_feats))
        vis_feats = vis_feats.flatten(start_dim=1, end_dim=2) #BxFxNxD -> Bx(FxN)xD
        vis_token_mask = torch.cat(vis_framewise_masks, dim=-1)
        
        if(self.cfg.VLM.USE_PRETRAINED_LANG_ENCODER):
            question_summary_rep, question_tokens_rep, question_tokens_length =  self._run_lang_encoder_transformer(in_x = lang_question_x,
                                                                                                                    lang_encoder_model = self.lang_encoder,
                                                                                                                    token_proj_layer = self.proj_lang_tokens,
                                                                                                                    summary_proj_layer = self.proj_lang_summary,
                                                                                                                      summary_drop_layer = self.lang_summary_rep_drop,
                                                                                                                    device = vis_feats.device
                                                                                                                     )
            option0_summary_rep, option0_tokens_rep, option0_tokens_length =  self._run_lang_encoder_transformer(in_x = lang_choice0_x,
                                                                                                            lang_encoder_model = self.option_encoder,
                                                                                                            token_proj_layer = self.proj_option0_tokens,
                                                                                                            summary_proj_layer = self.proj_option0_summary,
                                                                                                            summary_drop_layer = self.option0_summary_rep_drop,
                                                                                                              device = vis_feats.device
                                                                                                                    
                                                                                                            )
            option1_summary_rep, option1_tokens_rep, option1_tokens_length =  self._run_lang_encoder_transformer(in_x = lang_choice1_x,
                                                                                                            lang_encoder_model = self.option_encoder,
                                                                                                            token_proj_layer = self.proj_option1_tokens,
                                                                                                            summary_proj_layer = self.proj_option1_summary,
                                                                                                            summary_drop_layer = self.option1_summary_rep_drop,
                                                                                                              device = vis_feats.device
                                                                                                                    
                                                                                                            )
            
            option2_summary_rep, option2_tokens_rep, option2_tokens_length =  self._run_lang_encoder_transformer(in_x = lang_choice2_x,
                                                                                                            lang_encoder_model = self.option_encoder,
                                                                                                            token_proj_layer = self.proj_option2_tokens,
                                                                                                            summary_proj_layer = self.proj_option2_summary,
                                                                                                            summary_drop_layer = self.option2_summary_rep_drop,
                                                                                                              device = vis_feats.device
                                                                                                                    
                                                                                                            )
            option3_summary_rep, option3_tokens_rep, option3_tokens_length =  self._run_lang_encoder_transformer(in_x = lang_choice3_x,
                                                                                                            lang_encoder_model = self.option_encoder,
                                                                                                            token_proj_layer = self.proj_option3_tokens,
                                                                                                            summary_proj_layer = self.proj_option3_summary,
                                                                                                            summary_drop_layer = self.option3_summary_rep_drop,
                                                                                                              device = vis_feats.device
                                                                                                                    
                                                                                                            )
            
            
            
            
        else:
            question_summary_rep, question_tokens_rep, question_tokens_length =  self._run_lang_encoder_lstm(in_x = lang_question_x,
                                                                                                                    embed_layer = self.embed,
                                                                                                                    lstm_block = self.lstm,
                                                                                                                    lang_summary_drop_layer = self.lang_summary_rep_drop,
                                                                                                                     token_proj_layer = self.proj_lang_tokens,
                                                                                                                    summary_proj_layer = self.proj_lang_summary,
                                                                                                                     device = vis_feats.device
                                                                                                                    
                                                                                                                   )
                
            option0_summary_rep, option0_tokens_rep, option0_tokens_length =  self._run_lang_encoder_lstm(in_x = lang_choice0_x,
                                                                                                            embed_layer = self.embed,
                                                                                                            lstm_block = self.option_lstm,
                                                                                                            lang_summary_drop_layer = self.option0_summary_rep_drop,
                                                                                                            token_proj_layer = self.proj_option0_tokens,
                                                                                                            summary_proj_layer =self.proj_option0_summary,
                                                                                                            device = vis_feats.device
                                                                                                                    
                                                                                                            )

            option1_summary_rep, option1_tokens_rep, option1_tokens_length =  self._run_lang_encoder_lstm(in_x = lang_choice1_x,
                                                                                                            embed_layer = self.embed,
                                                                                                            lstm_block = self.option_lstm,
                                                                                                            lang_summary_drop_layer = self.option1_summary_rep_drop,
                                                                                                          token_proj_layer = self.proj_option1_tokens,
                                                                                                            summary_proj_layer =self.proj_option1_summary,
                                                                                                           device = vis_feats.device
                                                                                                                    
                                                                                                            )
            option2_summary_rep, option2_tokens_rep, option2_tokens_length =  self._run_lang_encoder_lstm(in_x = lang_choice2_x,
                                                                                                            embed_layer = self.embed,
                                                                                                            lstm_block = self.option_lstm,
                                                                                                            lang_summary_drop_layer = self.option2_summary_rep_drop,
                                                                                                          token_proj_layer = self.proj_option2_tokens,
                                                                                                            summary_proj_layer =self.proj_option2_summary,
                                                                                                           device = vis_feats.device
                                                                                                                    
                                                                                                            )
            option3_summary_rep, option3_tokens_rep, option3_tokens_length =  self._run_lang_encoder_lstm(in_x = lang_choice3_x,
                                                                                                            embed_layer = self.embed,
                                                                                                            lstm_block = self.option_lstm,
                                                                                                            lang_summary_drop_layer = self.option3_summary_rep_drop,
                                                                                                          token_proj_layer = self.proj_option3_tokens,
                                                                                                            summary_proj_layer =self.proj_option3_summary,
                                                                                                           device = vis_feats.device
                                                                                                                    
                                                                                                            )
            
        if(self.cfg.VLM.ADD_LANG_AND_CHOICE_POSITIONAL_EMBEDS):
            question_tokens_rep = question_tokens_rep + self.question_lang_pos
            option0_tokens_rep = option0_tokens_rep + self.option0_lang_pos
            option1_tokens_rep = option1_tokens_rep + self.option1_lang_pos
            option2_tokens_rep = option2_tokens_rep + self.option2_lang_pos
            option3_tokens_rep = option3_tokens_rep + self.option3_lang_pos
        
        if(self.add_aux_lang_token):
            b = question_tokens_rep.size(0)
            question_tokens_rep = torch.cat([self.aux_lang_token.repeat(b, 1, 1), question_tokens_rep], dim=1)
            question_tokens_length = question_tokens_length + 1
        
        choice_outs = []
        for op_i in range(4):
            if(op_i==0):
                op_token_rep = option0_tokens_rep
                op_summ_rep = option0_summary_rep
                op_token_len = option0_tokens_length
            elif(op_i==1):
                op_token_rep = option1_tokens_rep
                op_summ_rep = option1_summary_rep
                op_token_len = option1_tokens_length
            if(op_i==2):
                op_token_rep = option2_tokens_rep
                op_summ_rep = option2_summary_rep
                op_token_len = option2_tokens_length
            if(op_i==3):
                op_token_rep = option3_tokens_rep
                op_summ_rep = option3_summary_rep
                op_token_len = option3_tokens_length
                
            choice_outs.append(self.run_individual_choice(question_tokens_rep, 
                                                          op_token_rep,
                                                          question_tokens_length,
                                                          op_token_len,
                                                          question_summary_rep,
                                                          op_summ_rep,
                                                          vis_feats,
                                                          vis_token_mask)
                              )
            
                                                          
        if(self.cfg.VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE in ['joint_concat']):
            final_out = self.concat_classifier(torch.cat(choice_outs, dim=-1))
        elif(self.cfg.VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE in ['individual']):
            final_outs = []
            final_outs.append(self.indv_classifier(choice_outs[0]))
            final_outs.append(self.indv_classifier(choice_outs[1]))
            final_outs.append(self.indv_classifier(choice_outs[2]))
            final_outs.append(self.indv_classifier(choice_outs[3]))
            final_out = torch.cat(final_outs, dim=-1)
            
        return final_out
            
    def run_individual_choice(self, 
                             q_token_rep, 
                             o_tokens_rep, 
                             q_tokens_length,
                             o_tokens_length,
                             q_summary_rep,
                             o_summary_rep,
                             v_feats,
                             v_token_mask,
                            ):
        if(self.cfg.VLM.CONCAT_QUESTION_AND_OPTION_LANG_REPS_FOR_MODULE):
            cumulative_lang_rep = torch.cat([q_token_rep, o_tokens_rep], dim=1) #-> Bx(N1+N2)xD  
            q_mask = length_to_mask(q_tokens_length).to(cumulative_lang_rep)
            o_mask = length_to_mask(o_tokens_length).to(cumulative_lang_rep)
            cumulative_lang_mask = torch.cat([q_mask, o_mask], dim=-1)
        else:
            cumulative_lang_rep = q_token_rep
            cumulative_lang_mask = length_to_mask(q_tokens_length).to(cumulative_lang_rep)
        
        if(self.cfg.VLM.COMBINE_QUES_AND_OPTION_SUMMARY_FOR_MODULE):
            combined_lang_option_summary = self.combine_ques_and_option_summary_proj(torch.cat([q_summary_rep, o_summary_rep], dim=-1))
            combined_lang_option_summary = self.combine_ques_and_option_summary_drop(combined_lang_option_summary)
        else:
            combined_lang_option_summary = q_summary_rep
            
        module_output = self.vlm_module(cumulative_lang_rep, #B x (Nq+Nopt) x D
                                        combined_lang_option_summary, #B x D
                                        v_feats.transpose(-1,-2), #->BxDx(FxN)
                                        cumulative_lang_mask,
                                        v_token_mask,
                                        save_atts =False
                                       )
        
        if(self.cfg.VLM.CLASSIFIER_TYPE in ['only_result']):
            out = self.classifier(module_output)
        else:
            ques_summary_rep_preclassify = self.lang_summary_rep_pre_classifier_proj(q_summary_rep)
            opt_summary_rep_preclassify = self.option_summary_rep_pre_classifier_proj(o_summary_rep)
            out = torch.cat([module_output, ques_summary_rep_preclassify, opt_summary_rep_preclassify], dim=1)
            out = self.classifier(out)

        return out
        
