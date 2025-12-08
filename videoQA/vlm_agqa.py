import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from einops import rearrange
from torch.nn.init import trunc_normal_
from basic_model_blocks import get_activation_layer, linear, length_to_mask, Transformer_Self_Attention_Block
import copy
class VLM_for_AGQA(nn.Module):
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
        self.obj_label_embed = torch.load(cfg.VLM.OBJ_LABEL_EMBD_LAYER_PATH)
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
            self.tokenizer = torch.load(cfg.VLM.PRETRAINED_LANG_TOKENIZER_PATH)
            self.lang_encoder = torch.load(cfg.VLM.PRETRAINED_LANG_ENCODER_PATH)
            lang_enc_dim = self.lang_encoder.embeddings.word_embeddings.weight.size(-1)
            self.proj_lang_summary = linear(lang_enc_dim, dim)
            self.proj_lang_tokens = linear(lang_enc_dim, dim)
            if(self.cfg.VLM.SHARE_QUESTION_AND_OPTION_ENCODER):
                self.option_encoder = self.lang_encoder
            else:
                self.option_encoder = copy.deepcopy(self.lang_encoder)
            
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
        self.option_summary_rep_pre_classifier_proj = linear(dim, dim)
        
            
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
                lang_question_x, #(lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x), #this tuple = lang_x
                objs_resolutions, #of shape BxFx2 (used to normalize bbox)
                num_objs, #of shape BxF (list used to make mask)
                num_unpadded_frames, #of shape B (list used to make mask)
                save_atts=False):
        person_bbox_x, person_keypoints_x = person_vis_x_tuple
        person_keypoints_x = person_keypoints_x.squeeze(2) #B x F x 1 x N_k x 3 -> B x F x N_k x 3
        person_keypoints_x = person_keypoints_x[:,:,:,:2] #B x F x N_k x 3 -> #B x F x N_k x 2
        obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x = objs_vis_x_tuple
        lang_question_x
        
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
            
            
        else:
            question_summary_rep, question_tokens_rep, question_tokens_length =  self._run_lang_encoder_lstm(in_x = lang_question_x,
                                                                                                                    embed_layer = self.embed,
                                                                                                                    lstm_block = self.lstm,
                                                                                                                    lang_summary_drop_layer = self.lang_summary_rep_drop,
                                                                                                                     token_proj_layer = self.proj_lang_tokens,
                                                                                                                    summary_proj_layer = self.proj_lang_summary,
                                                                                                                     device = vis_feats.device
                                                                                                                    
                                                                                                                   )
                
            
            
        
        
        if(self.add_aux_lang_token):
            b = question_tokens_rep.size(0)
            question_tokens_rep = torch.cat([self.aux_lang_token.repeat(b, 1, 1), question_tokens_rep], dim=1)
            question_tokens_length = question_tokens_length + 1
        
        module_output = self.vlm_module(question_tokens_rep, 
                                        question_summary_rep,
                                        vis_feats.transpose(-1,-2),
                                        length_to_mask(question_tokens_length).to(question_tokens_rep),
                                        vis_token_mask,
                                        save_atts =False
                                       )
            
                                                          
        if(self.cfg.VLM.CLASSIFIER_TYPE in ['only_result']):
            out = self.classifier(module_output)
        else:
            question_summary_rep = self.lang_summary_rep_pre_classifier_proj(question_summary_rep)
            out = torch.cat([module_output, question_summary_rep], dim=1)
            out = self.classifier(out)
            
        return out
            
    
        
