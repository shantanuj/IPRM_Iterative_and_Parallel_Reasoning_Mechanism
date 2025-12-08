import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_model_blocks import get_activation_layer, Multi_Head_Attention_Block_Two_Values, linear, length_to_mask, Multi_Head_Attention_Block, linear2
from einops import rearrange


class Language_Retrieval_Block(nn.Module):
    def __init__(self, 
                 dim, 
                 lang_block_nonlin = 'tanh',
                 lang_block_mlp_drop = 0.1,
                 lang_block_attn_drop = 0.0,
                 use_prior_result_states_for_query = False,
                 prior_result_state_proj_nonlin = 'tanh',
                 do_concat_proj_for_joint_query=False,
                 #do_time_specific_proj=False,
                 #num_time_steps=6,
                 use_ln=False
                ):
        
        super().__init__()
        
        self.attn = linear(dim, 1)
        self.use_prior_result_states_for_query = use_prior_result_states_for_query
        #self.do_time_specific_proj = do_time_specific_proj
        
        self.lang_mlp = nn.Sequential(linear(dim, dim), 
                                          get_activation_layer(lang_block_nonlin), 
                                          nn.Dropout(lang_block_mlp_drop),
                                          linear(dim, dim)
                                         )
        if(self.use_prior_result_states_for_query):
            self.res_mlp = nn.Sequential(linear(dim, dim), 
                                          get_activation_layer(prior_result_state_proj_nonlin), 
                                          nn.Dropout(lang_block_mlp_drop),
                                          linear(dim, dim)
                                         )
            self.do_concat_proj_for_joint_query = do_concat_proj_for_joint_query
            if(self.do_concat_proj_for_joint_query):
                self.joint_proj = linear(2*dim, dim)
            
                                      
        self.attn_drop = nn.Dropout(lang_block_attn_drop)
        self.ln_lang = nn.LayerNorm(dim) if use_ln else nn.Identity()
        
        
    def forward(
        self,
        step,
        lang_tokens_rep, #BxNxD
        lang_summary_rep, #BxKxD where K is num operation tokens
        lang_token_mask,
        prev_op_state,
        prior_result_state,
        lang_tokens_rep_lens=None,
        ):
        b, l, c = lang_tokens_rep.shape
        assert(lang_tokens_rep.ndim==3) #BxNxD (where N is num lang tokens)
        query = prev_op_state
        assert(query.ndim==3)
            
        lang_summary_rep = self.lang_mlp(query)
        
        if(self.use_prior_result_states_for_query):
            if(self.do_concat_proj_for_joint_query):
                lang_summary_rep = self.joint_proj(torch.cat([lang_summary_rep, self.res_mlp(prior_result_state)], dim=-1))
            else:
                lang_summary_rep = lang_summary_rep + self.res_mlp(prior_result_state)
            

        
        lang_tokens_rep_prod = lang_summary_rep.unsqueeze(2) * lang_tokens_rep.unsqueeze(1)
        attn_weight = self.attn(lang_tokens_rep_prod)
        
        attn_mask = lang_token_mask
        attn_mask = 1.0 - attn_mask
        attn_mask = attn_mask.float().unsqueeze(1).unsqueeze(3)
        attn_weight = attn_weight + (attn_mask * -1e30)

        attn = F.softmax(attn_weight, dim=2) #bxkxnx1 
        attn = self.attn_drop(attn)
        retrieved_lang = (attn * lang_tokens_rep.unsqueeze(1)).sum(dim=2) #bxkxnx1 X bx1xnxd -> bxkxnxd

        return self.ln_lang(retrieved_lang), attn_weight

                
    
class Visual_Retrieval_Block(nn.Module):
    def __init__(self, 
                 dim, 
                 vis_block_drop=0.1,
                 vis_block_attn_drop = 0.0,
                 vis_block_nonlin = 'elu',
                 reduction_r = 1,
                 use_ln=True
                ):
        super().__init__()

        self.dropout_vis = nn.Dropout(vis_block_drop)
        self.dropout_prev_res = nn.Dropout(vis_block_drop)
        self.dropout_lang = nn.Dropout(vis_block_drop)
        
        red_dim = dim//reduction_r
        self.reduction_r = reduction_r
        self.proj_vis = linear(dim, red_dim)
        self.proj_vis2 = linear(dim, dim)
        #if(reduction_r == 1):
          #  self.proj_vis2 = None #self.proj_vis
        self.proj_concat_mod_and_og_vis = linear2(red_dim * 2, red_dim, nonlin= vis_block_nonlin)
        self.proj_prev_res = linear(dim, red_dim)
        self.proj_lang = linear(dim, red_dim)
        self.proj_lang2 = linear(dim, red_dim)
            
             
        self.inter2att = linear(red_dim, 1)
        self.inter_dropout = nn.Dropout(vis_block_drop)
        self.attn_drop = nn.Dropout(vis_block_attn_drop)
        
        self.nonlin = get_activation_layer(vis_block_nonlin)
        self.ln_vis = nn.LayerNorm(dim) if use_ln else nn.Identity()
        

    def gen_dropout_mask(self, prev_res, vis_dropout=0.15):
        mask = torch.empty_like(prev_res).bernoulli_(1 - vis_dropout)
        mask = mask / (1 - vis_dropout)
        self.memory_dropout_mask = mask

    
    
    def forward(self, prev_result, vis_tokens_rep, lang_rep, prev_op_state, vis_token_mask): #vis_token_lens, vis_tokens_lens_max):
        if self.training:
            prev_result = self.memory_dropout_mask * prev_result

        assert(prev_result.ndim==3) #BxKxD
        vis_rep_dropped = self.dropout_vis(vis_tokens_rep)
        prev_res = self.dropout_prev_res(prev_result)

        vis_rep_dropped = rearrange(vis_rep_dropped, "b c hw -> b hw c") #bxnxd
        vis_rep = self.proj_vis(vis_rep_dropped)
        feat_mod_wts = self.proj_prev_res(prev_res) + self.proj_lang(self.dropout_lang(lang_rep))
            
            

        feat_mod_wts = feat_mod_wts.unsqueeze(2) #bxkxd -> bxkx1xd
        out = feat_mod_wts * vis_rep.unsqueeze(1) #bxkx1xd X bx1xnxd -> bxkxnxd
        
        inter = torch.cat([out, vis_rep.unsqueeze(1).repeat(1,feat_mod_wts.size(1),1,1)], dim=3)
        inter = self.proj_concat_mod_and_og_vis(inter) #bxkxnxd
        
        inter = inter * self.proj_lang2(lang_rep).unsqueeze(2) #bxkxnxd X bxkx1xd - > bxkxnxd
        inter = self.nonlin(inter)
        
        inter = self.inter_dropout(inter)
        attention_weight = self.inter2att(inter)
        
        #attn_mask = length_to_mask(vis_token_lens, max_len=vis_tokens_lens_max).to(attention_weight) #bxn
        attn_mask = 1.0 - vis_token_mask
        attn_mask = attn_mask.float().unsqueeze(1).unsqueeze(3) #bx1xnx1

        attention_weight = attention_weight + (attn_mask * -1e30) 

        attention = F.softmax(attention_weight, dim=2) #bxkxnx1 (softmax along 2nd index)
        attention = self.attn_drop(attention)
        #if(not self.reduction_r==1):
        retrieved_vis = (attention * self.proj_vis2(vis_rep_dropped).unsqueeze(1)).sum(dim=2) #bxkxnx1 X bx1xnxd
        #else:
         #   retrieved_vis = (attention * vis_rep.unsqueeze(1)).sum(dim=2) #bxkxnx1 X bx1xnxd
        return self.ln_vis(retrieved_vis), attention_weight

class Simpler_Visual_Retrieval_Block(nn.Module):
    def __init__(self, 
                 dim, 
                 vis_block_drop=0.1,
                 vis_block_attn_drop = 0.0,
                 vis_block_nonlin = 'elu',
                 use_mlp_for_prior_results_proj=False,
                 use_mlp_for_lang_proj=False,
                 lang_proj_nonlin='elu',
                 reduction_r = 1,
                 use_ln=False,
                 do_concat_proj_for_joint_query=False
                ):
        super().__init__()

        self.dropout_vis = nn.Dropout(vis_block_drop)
        self.dropout_prev_res = nn.Dropout(vis_block_drop) if not use_mlp_for_prior_results_proj else nn.Identity()
        self.dropout_lang = nn.Dropout(vis_block_drop) if not use_mlp_for_lang_proj else nn.Identity()
        
        red_dim = dim//reduction_r
        self.reduction_r = reduction_r

        self.v_proj = linear(dim, dim)
        self.k_proj = linear2(dim, red_dim, nonlin=vis_block_nonlin)
        if(use_mlp_for_prior_results_proj):
            self.proj_prev_res = nn.Sequential(linear(dim, red_dim), 
                                          get_activation_layer('tanh'), 
                                          nn.Dropout(vis_block_drop),
                                          linear(red_dim, red_dim)
                                         )
            #linear2(dim, red_dim, nonlin=vis_block_nonlin)
        else:
            self.proj_prev_res = linear(dim,  red_dim)
        if(use_mlp_for_lang_proj):
            self.proj_lang = nn.Sequential(linear(dim, red_dim), 
                                          get_activation_layer('tanh'), 
                                          nn.Dropout(vis_block_drop),
                                          linear(red_dim, red_dim)
                                         )
            #self.proj_lang = linear2(dim, red_dim, nonlin=lang_proj_nonlin)
        else:
            self.proj_lang = linear(dim, red_dim)

        self.do_concat_proj_for_joint_query = do_concat_proj_for_joint_query
        if(self.do_concat_proj_for_joint_query):
            self.query_proj = linear(2*red_dim, red_dim)
        #else:
         #   self.query_pr
        
        #self.proj_lang2 = linear(dim, dim)
        
        #self.proj_vis = linear(dim, red_dim)
        #self.proj_vis2 = linear(dim, dim)
        #if(reduction_r == 1):
          #  self.proj_vis2 = None #self.proj_vis
        #self.proj_concat_mod_and_og_vis = linear2(red_dim * 2, red_dim, nonlin= vis_block_nonlin)
        #self.proj_prev_res = linear(dim, red_dim)
        #self.proj_lang = linear(dim, red_dim)
        #self.proj_lang2 = linear(dim, red_dim)
            
             
        self.inter2att = linear(red_dim, 1)
        self.inter_dropout = nn.Dropout(vis_block_drop)
        self.attn_drop = nn.Dropout(vis_block_attn_drop)
        
        self.nonlin = get_activation_layer(vis_block_nonlin)
        self.ln_vis = nn.LayerNorm(dim) if use_ln else nn.Identity()
        

    def gen_dropout_mask(self, prev_res, vis_dropout=0.15):
        mask = torch.empty_like(prev_res).bernoulli_(1 - vis_dropout)
        mask = mask / (1 - vis_dropout)
        self.memory_dropout_mask = mask

    
    
    def forward(self, prev_result, vis_tokens_rep, lang_rep, prev_op_state, vis_token_mask): #vis_token_lens, vis_tokens_lens_max):
        if self.training:
            prev_result = self.memory_dropout_mask * prev_result

        assert(prev_result.ndim==3) #BxKxD
        vis_rep_dropped = self.dropout_vis(vis_tokens_rep)
        prev_res = self.dropout_prev_res(prev_result)

        vis_rep_dropped = rearrange(vis_rep_dropped, "b c hw -> b hw c") #bxnxd
        vis_key = self.k_proj(vis_rep_dropped)
        if(self.do_concat_proj_for_joint_query):
            query = self.query_proj(torch.cat([self.proj_lang(self.dropout_lang(lang_rep)), self.proj_prev_res(prev_res)], dim=-1))
        else:
            query = self.proj_lang(self.dropout_lang(lang_rep)) + self.proj_prev_res(prev_res)
            
        #feat_mod_wts = self.proj_prev_res(prev_res) + self.proj_lang(self.dropout_lang(lang_rep))
            
            

        #feat_mod_wts = feat_mod_wts.unsqueeze(2) #bxkxd -> bxkx1xd
        #out = feat_mod_wts * vis_rep.unsqueeze(1) #bxkx1xd X bx1xnxd -> bxkxnxd
        
        #inter = torch.cat([out, vis_rep.unsqueeze(1).repeat(1,feat_mod_wts.size(1),1,1)], dim=3)
        #inter = self.proj_concat_mod_and_og_vis(inter) #bxkxnxd
        
        inter = query.unsqueeze(2) * vis_key.unsqueeze(1) #bxkx1xd x bx1xnxd - > bxkxnxd
        inter = self.nonlin(inter)
        
        inter = self.inter_dropout(inter)
        attention_weight = self.inter2att(inter) #bxkxnx1
        
        #attn_mask = length_to_mask(vis_token_lens, max_len=vis_tokens_lens_max).to(attention_weight) #bxn
        attn_mask = 1.0 - vis_token_mask
        attn_mask = attn_mask.float().unsqueeze(1).unsqueeze(3) #bx1xnx1

        attention_weight = attention_weight + (attn_mask * -1e30) 

        attention = F.softmax(attention_weight, dim=2) #bxkxnx1 (softmax along 2nd index)
        attention = self.attn_drop(attention)
        #if(not self.reduction_r==1):
        retrieved_vis = (attention * self.v_proj(vis_rep_dropped).unsqueeze(1)).sum(dim=2) #bxkxnx1 X bx1xnxd
        #else:
         #   retrieved_vis = (attention * vis_rep.unsqueeze(1)).sum(dim=2) #bxkxnx1 X bx1xnxd
        return self.ln_vis(retrieved_vis), attention_weight

class Memory_Update_Block(nn.Module):
    def __init__(self, 
                 dim,
                 use_ln=False,
                 op_memupdate_nonlin='identity',
                 res_memupdate_nonlin='identity',
                ):
        super().__init__()
        
        self.proj_new_result_state = linear(dim*2, dim)    
        self.ln_res = nn.LayerNorm(dim) if use_ln else nn.Identity()
        
        self.proj_new_lang_state = linear(dim*2, dim)
        self.ln_op = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.op_memory_nonlin = get_activation_layer(op_memupdate_nonlin)
        self.op_res_nonlin = get_activation_layer(res_memupdate_nonlin)
            
    def forward(self, result_states, retrieved_vis, op_states, retrieved_lang):
        retrieved_lang = self.op_memory_nonlin(retrieved_lang)
        new_op_state = self.proj_new_lang_state(torch.cat([retrieved_lang, op_states[-1]], dim=-1))
        #new_op_state = self.op_memory_nonlin(new_op_state)
        new_op_state = self.ln_op(new_op_state)
        op_states.append(new_op_state)

        retrieved_vis = self.op_res_nonlin(retrieved_vis)
        new_res_state = self.proj_new_result_state(torch.cat([result_states[-1], retrieved_vis], dim=-1))
        #new_res_state = self.op_res_nonlin(new_res_state)
        new_res_state = self.ln_res(new_res_state)
        result_states.append(new_res_state)
        return result_states, op_states

class Operation_Interaction_Unit(nn.Module):
    def __init__(self, 
                 dim, 
                 mask_self_in_attn=False,
                 attn_mod_drop = 0.1,
                 update_res_drop = 0.1,
                 attn_matrix_drop = 0.0,
                 key_proj_of_lang_state=False,
                 use_ln=True
                ):
        super().__init__()
                
        self.op_state_query = linear(dim, dim)
        self.op_state_key = linear(dim, dim) if key_proj_of_lang_state else nn.Identity()
        self.attn = linear(dim, 1)
        self.op_state_value = linear(dim, dim)
        self.result_state_value = linear(dim, dim)
        self.weighted_residual_op_state = linear(dim, dim)
        self.weighted_residual_result_state = linear(dim, dim)
        
        self.mask_self_in_attn = mask_self_in_attn 
        self.mod_drop = nn.Dropout(attn_mod_drop)
        self.result_update_drop = nn.Dropout(update_res_drop)
        self.op_update_drop = nn.Dropout(update_res_drop)
        
        self.attn_drop = nn.Dropout(attn_matrix_drop)
        
        self.ln_op_post = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.ln_res_post = nn.LayerNorm(dim) if use_ln else nn.Identity()
        
        
            
    def forward(self, 
                op_states,#BxNxD
                result_states, #BxNxD
                queue_op_states = None, #list of TxBxNxD
                queue_result_states = None, #list of TxBxNxD
               ):
        q_op_state = self.op_state_query(op_states)
        windowed_att = False
        if(queue_op_states is None or len(queue_op_states)<1):
            k_op_state = self.op_state_key(op_states)
            v_op_state = self.op_update_drop(self.op_state_value(op_states))
            v_result_state = self.result_update_drop(self.result_state_value(result_states))
        else:
            assert(queue_result_states is not None and len(queue_result_states)==len(queue_op_states))
            windowed_att = True
            queue_op_states_stacked = torch.stack(queue_op_states) #TxBxNxD
            queue_op_states_stacked = rearrange(queue_op_states_stacked, "t b n d -> b (t n) d") #B x WN x D
            op_states_window = torch.cat([op_states, queue_op_states_stacked], dim=1)
            k_op_state = self.op_state_key(op_states_window)
            v_op_state = self.op_update_drop(self.op_state_value(op_states_window))
            
            queue_result_states_stacked = torch.stack(queue_result_states) #TxBxNxD
            queue_result_states_stacked = rearrange(queue_result_states_stacked, "t b n d -> b (t n) d") #B x WN x D
            res_states_window = torch.cat([result_states, queue_result_states_stacked], dim=1)
            v_result_state = self.result_update_drop(self.result_state_value(res_states_window))
            
            
        attn_prod = q_op_state.unsqueeze(2) * k_op_state.unsqueeze(1) #bxNx1xd x bx1xNxD
        attn_prod = self.mod_drop(attn_prod) 
        attn_weight = self.attn(attn_prod) #bxNxNx1
        
        if(self.mask_self_in_attn):
            b, n, d = op_states.shape
            if(windowed_att):
                n_w = v_result_state.size(1)
                attn_mask = torch.eye(n_w, dtype=torch.bool)[:n].unsqueeze(0).expand(b, -1, -1)
                #print(attn_mask)
                #input()
            else:
                attn_mask = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
            
            attn_mask = attn_mask.to(attn_prod)
            attn_mask = attn_mask.float().unsqueeze(3) #bxNxNx1
            attn_weight = attn_weight + (attn_mask * -1e30)
            
        attn_weight = F.softmax(attn_weight, dim=2) #BxNxNx1
        #print(attn_weight)
        #input()
            
        attn_weight = self.attn_drop(attn_weight)
        
        new_op_state = (attn_weight * v_op_state.unsqueeze(1)).sum(dim=2)   #BxNxNx1 x Bx1xNxD -> BxNxNxD 
        new_op_state = new_op_state + self.weighted_residual_op_state(op_states)
        
        
        new_result_state = (attn_weight * v_result_state.unsqueeze(1)).sum(dim=2)
        new_result_state = new_result_state + self.weighted_residual_result_state(result_states)
        
        return self.ln_op_post(new_op_state), self.ln_res_post(new_result_state), attn_weight
            
        

class Recurrent_Memory_Attention_Module(nn.Module):
    def __init__(self, 
                 dim,
                 cfg, 
                ):
        super().__init__()
        self.cfg = cfg
        self.dim = dim = cfg.VLM.DIM
        self.max_computation_steps = cfg.IPRM.NUM_COMPUTATION_STEPS
        self.lang_retrieval_block = Language_Retrieval_Block(dim, 
                                                             lang_block_nonlin = cfg.IPRM.LANG_BLOCK_NONLIN,
                                                             lang_block_mlp_drop = cfg.IPRM.LANG_BLOCK_MLP_DROP,
                                                             lang_block_attn_drop = cfg.IPRM.LANG_BLOCK_ATTN_DROP,
                                                             use_ln = cfg.IPRM.LANG_BLOCK_USE_LN,
                                                             use_prior_result_states_for_query = cfg.IPRM.USE_PRIOR_RESULT_STATES_FOR_LANG_QUERY,
                                                             prior_result_state_proj_nonlin = cfg.IPRM.PRIOR_RESULT_STATE_MLP_NONLIN_FOR_LANG_RETRIEVAL,
                                                             do_concat_proj_for_joint_query = cfg.IPRM.DO_CONCAT_PROJ_FOR_QUERY_LANG_RETRIEVAL
                                                             
                                                             )
        if(cfg.IPRM.USE_SIMPLER_VIS_RETRIEVAL_BLOCK):
            self.vis_retrieval_block = Simpler_Visual_Retrieval_Block(dim,
                                                          vis_block_drop= cfg.IPRM.VIS_BLOCK_DROP,
                                                          vis_block_attn_drop = cfg.IPRM.VIS_BLOCK_ATTN_DROP,
                                                          vis_block_nonlin = cfg.IPRM.VIS_BLOCK_NONLIN,
                                                          reduction_r = cfg.IPRM.REDUCTION_R,
                                                          use_ln = cfg.IPRM.VIS_BLOCK_USE_LN,
                                                                      use_mlp_for_prior_results_proj= cfg.IPRM.USE_MLP_FOR_PRIOR_RESULTS_PROJ_IN_VISRETRIEVAL,
                                                         use_mlp_for_lang_proj=cfg.IPRM.USE_MLP_FOR_LANG_PROJ_IN_VISRETRIEVAL,
                                                         lang_proj_nonlin=cfg.IPRM.NONLIN_FOR_LANG_PROJ_IN_VISRETRIEVAL,
                                                          do_concat_proj_for_joint_query = cfg.IPRM.DO_CONCAT_PROJ_FOR_QUERY_VIS_RETRIEVAL
                                                          )
        else:
            
            self.vis_retrieval_block = Visual_Retrieval_Block(dim,
                                                          vis_block_drop= cfg.IPRM.VIS_BLOCK_DROP,
                                                          vis_block_attn_drop = cfg.IPRM.VIS_BLOCK_ATTN_DROP,
                                                          vis_block_nonlin = cfg.IPRM.VIS_BLOCK_NONLIN,
                                                          reduction_r = cfg.IPRM.REDUCTION_R,
                                                          use_ln = cfg.IPRM.VIS_BLOCK_USE_LN,
                                                         
                                                         
                                                          )
        
        if(cfg.IPRM.DO_MEMORY_UPDATE):
            self.memory_update_block = Memory_Update_Block(dim,
                                                           use_ln = cfg.IPRM.MEM_UPDATE_BLOCK_USE_LN,
                                                           op_memupdate_nonlin = cfg.IPRM.MEM_OP_UPDATE_NONLIN,
                                                           res_memupdate_nonlin = cfg.IPRM.MEM_RES_UPDATE_NONLIN
                                                          )
        #else:
         #   cfg.IPRM.DO_MEMORY_OP_INTERACTION = False #set mem op interaction to false as well
        if(cfg.IPRM.DO_MEMORY_OP_INTERACTION):
            self.memory_interaction_block = Operation_Interaction_Unit(dim, 
                                                                       mask_self_in_attn=cfg.IPRM.MEMORY_STATE_UPDATE_MASK_SELF_IN_ATTN,
                                                                       attn_mod_drop = cfg.IPRM.MEMORY_STATE_UPDATE_MOD_DROPOUT,
                                                                       update_res_drop = cfg.IPRM.MEMORY_STATE_UPDATE_RES_DROPOUT,
                                                                       attn_matrix_drop = cfg.IPRM.MEM_ATTN_MATRIX_DROP,
                                                                       key_proj_of_lang_state = cfg.IPRM.MEMORY_STATE_UPDATE_DO_KEY_PROJ_OF_LANG_STATE,
                                                                       use_ln = cfg.IPRM.OP_INTERACTION_BLOCK_USE_LN
                                                                      )
            
        
        self.num_memory_tokens = cfg.IPRM.NUM_MEMORY_TOKENS
        
        if(cfg.IPRM.MEMORY_OP_STATE_INIT in ['random']):
            self.op_states_0 = nn.Parameter(torch.randn(self.num_memory_tokens, dim))
        elif(cfg.IPRM.MEMORY_OP_STATE_INIT in ['lang_projs']):
            self.op_init_projs = nn.ModuleList([linear(dim, dim) for i in range(self.num_memory_tokens)])
        else:
            None
            
        if(cfg.IPRM.MEMORY_RES_STATE_INIT in ['random']):
            self.res_states_0 = nn.Parameter(torch.randn(self.num_memory_tokens, dim))
        elif(cfg.IPRM.MEMORY_RES_STATE_INIT in ['vis_projs_on_avp']):
            self.res_init_projs = nn.ModuleList([linear(dim, dim) for i in range(self.num_memory_tokens)])
        else:
            None
            
        
        if(cfg.IPRM.MEMORY_RES_POOL_METHOD in ['random_token_query_pool_op']):
            self.random_token_result_pool = nn.Parameter(torch.randn(1, 1, dim))
            self.result_pool_mod_attn_layer = linear(dim, 1)
            self.pool_k_linear_layer = linear(dim, dim)
            
        elif(cfg.IPRM.MEMORY_RES_POOL_METHOD in ['lang_cuml_query_op']):
            self.result_pool_mod_attn_layer = linear(dim, 1)
            self.pool_q_linear_layer = linear(dim, dim)
            self.pool_k_linear_layer = linear(dim, dim)
        else:
            None
            
        self.attentions = {"text": None, "image": None, "inter": None, "start": None}
        self.add_aux_lang_token = cfg.IPRM.ADD_AUX_LANG_TOKEN
        if(self.add_aux_lang_token):
            self.aux_lang_token = nn.Parameter(torch.randn(1, 1, dim))
            
        self.memory_window_len = cfg.IPRM.MEMORY_ATT_WINDOW_LEN
        
        
    def _init_memory(self, 
                     visual_tokens, #BxDxNv
                     lang_tokens, #BxNlxD
                     lang_cuml=None #BxD
                    ):
        
        b = visual_tokens.size(0)
        if(self.cfg.IPRM.MEMORY_RES_STATE_INIT in ['random']):
            res_states = self.res_states_0.expand(b, self.num_memory_tokens, self.dim)
            
        elif(self.cfg.IPRM.MEMORY_RES_STATE_INIT in ['vis_projs_on_avp']):
            vis_gist = visual_tokens.mean(dim=2) 
            res_states = torch.stack([res_proj(vis_gist) for res_proj in self.res_init_projs], dim=0)
            res_states = res_states.transpose(0,1) #Nm x B xD -> BxNmxD
        else:
            print(f"ERROR: MEMORY_RES_STATE_INIT method {self.cfg.IPRM.MEMORY_RES_STATE_INIT} not implemented yet")
            sys.exit(0)
            
        if(self.cfg.IPRM.MEMORY_OP_STATE_INIT in ['random']):
            op_states = self.op_states_0.expand(b, self.num_memory_tokens, self.dim)
        elif(self.cfg.IPRM.MEMORY_OP_STATE_INIT in ['lang_projs']):
            assert(lang_cuml is not None)
            op_states = torch.stack([op_proj(lang_cuml) for op_proj in self.op_init_projs], dim=0)
            op_states = op_states.transpose(0,1) #Nm x B xD -> BxNmxD
            
        else:
            print(f"ERROR: MEMORY_OP_STATE_INIT method {self.cfg.IPRM.MEMORY_OP_STATE_INIT} not implemented yet")
            sys.exit(0)
            
        return op_states, res_states
            
    def _do_modulated_attn(self, 
                           q, #BxNqxD
                           k, #BxNkxD
                           v, #BxNvxD
                           mod_attn_linear, #linear layer
                           q_linear=None,#=nn.Identity(),
                           k_linear=None,#=nn.Identity(),
                           v_linear=None,
                           #out_linear=None,
                           mod_drop_layer=None,
                           out_linear_layer=None,
                           out_drop_layer=None,
                           attn_drop_layer=None,
                           attn_mask=None, #0s where keep and 1s where mask; of shape BxNqxNk
                           attn_activation_fn='softmax'
                          ):
        if(q_linear is not None):
            q = q_linear(q)
        if(k_linear is not None):
            k = k_linear(k)
        if(v_linear is not None):
            v = v_linear(v)
        #if(out_drop_layer is not None):
        attn_prod = q.unsqueeze(2) * k.unsqueeze(1) #bxNqx1xD x bx1xNkxD -> B x Nq x Nk x D 
        if(mod_drop_layer is not None):
            attn_prod = mod_drop_layer(attn_prod) 
        attn_weight = mod_attn_linear(attn_prod) #bxNqxNkx1
            
        
        
        
        if(attn_mask is not None):
            attn_mask = attn_mask.to(attn_prod)
            attn_mask = attn_mask.float().unsqueeze(3) #bxNqxNvx1
            attn_weight = attn_weight + (attn_mask * -1e30)
            
        if(attn_activation_fn in ['sigmoid']):
            attn_weight = F.sigmoid(attn_weight)
        elif(attn_activation_fn in ['softmax']):
            attn_weight = F.softmax(attn_weight, dim=2) #BxNqxNvx1
        else:
            print(f"error: attn_activation_fn {attn_activation_fn} not found")
            sys.exit(0)
            
        out = (attn_weight * v.unsqueeze(1)).sum(dim=2)   #BxNqxNkx1 x Bx1xNvxD -> BxNqxNvxD -> BxNqxD (after sum)
        if(out_drop_layer is not None):
            out = out_drop_layer(out)
        if(out_linear_layer is not None):
            out = out_linear_layer(out)
            
        return out, attn_weight
            
        
                           
    def _pool_final_result(self, 
                           res_state, #B x Nm x D 
                           op_state,  #B x Nm x D
                           lang_cuml=None):
        b = res_state.size(0)
        attn_wt = None
        if(self.cfg.IPRM.MEMORY_RES_POOL_METHOD in ['avp']):
            return res_state.mean(dim=1), attn_wt
        elif(self.cfg.IPRM.MEMORY_RES_POOL_METHOD in ['first_token']):
            return res_state[:,0,:], attn_wt
        elif(self.cfg.IPRM.MEMORY_RES_POOL_METHOD in ['random_token_query_pool_op']):
            pooled_res, attn_wt = self._do_modulated_attn(q= self.random_token_result_pool.repeat(b,1,1),  #Bx1xD
                                                 k=op_state,
                                                 v=res_state,
                                                 mod_attn_linear=self.result_pool_mod_attn_layer,
                                                 k_linear=self.pool_k_linear_layer,
                                                )
            #output is Bx1xD
            return pooled_res.squeeze(1), attn_wt
        elif(self.cfg.IPRM.MEMORY_RES_POOL_METHOD in ['lang_cuml_query_op']):
            pooled_res, attn_wt = self._do_modulated_attn(q= lang_cuml.unsqueeze(1),  #Bx1xD
                                                 k=op_state,
                                                 v=res_state,
                                                 
                                                 mod_attn_linear=self.result_pool_mod_attn_layer,
                                                 q_linear = self.pool_q_linear_layer,
                                                 k_linear=self.pool_k_linear_layer,
                                                )
            return pooled_res.squeeze(1), attn_wt
                                                 
            
        else:
            print(f"Error: MEMORY_RES_POOL_METHOD: {self.cfg.IPRM.MEMORY_RES_POOL_METHOD} not found ")
            sys.exit(0)
        
    def forward(
        self,
        lang_tokens_rep, #B x Nl x D
        lang_summary_rep, #B x D
        vis_tokens_rep, #Bx D x Nv
        lang_token_mask,
        vis_token_mask,
        lang_tokens_rep_lens=None,
        vis_tokens_lens=None,
        vis_tokens_lens_max=None,
        save_atts =False
    ):
        self.attentions = {"text": None, "image": None, "inter": None, "start": None, 'inter_mem_op':None, 'final_pool_attn_wt':None}
        b_size = lang_summary_rep.size(0)
        
        
        #1. Initialize memory
        op_state, res_state = self._init_memory(vis_tokens_rep, lang_tokens_rep, lang_summary_rep)
            
        #2. dropout generation for prior result state  
        self.vis_retrieval_block.gen_dropout_mask(res_state)

        op_states = [op_state]
        res_states = [res_state]
        
        spatial_attentions = list()
        text_attentions = list()
        inter_mem_op_atts = list()

        for i in range(self.max_computation_steps):
            retrieved_lang, text_attention = self.lang_retrieval_block(
                i,
                lang_tokens_rep,
                op_state,
                lang_token_mask,
                prev_op_state = op_states[-1],
                prior_result_state = res_states[-1]
            )
            
            text_attentions.append(text_attention)
            retrieved_vis, attention = self.vis_retrieval_block(res_state, vis_tokens_rep, retrieved_lang, op_state, vis_token_mask=vis_token_mask)
            spatial_attentions.append(attention)
            
            if(self.cfg.IPRM.DO_MEMORY_UPDATE):
                res_states, op_states = self.memory_update_block(res_states, retrieved_vis, op_states, retrieved_lang)
            else:
                res_states.append(retrieved_vis)
                op_states.append(op_state)
                
            op_state = op_states.pop()
            res_state = res_states.pop()
            if(self.cfg.IPRM.DO_MEMORY_OP_INTERACTION):
                if(self.memory_window_len>0):
                    op_state, res_state, inter_mem_op_att = self.memory_interaction_block(op_state, res_state,
                                                                                          queue_op_states = op_states[-self.memory_window_len:][::-1], #list of TxBxNxD
                                                                                          queue_result_states = res_states[-self.memory_window_len:][::-1]
                                                                                         )
                else:    
                    op_state, res_state, inter_mem_op_att = self.memory_interaction_block(op_state, res_state)
                inter_mem_op_atts.append(inter_mem_op_att)
                
            op_states.append(op_state)
            res_states.append(res_state)
            op_states = op_states[min(-1,-self.memory_window_len):] #keep atleast the last unless more needed
            res_states = res_states[min(-1,-self.memory_window_len):] #keep atleast the last unless more needed
            #print(len(op_states), len(res_states))
        
        final_res_state, pool_attn_wt = self._pool_final_result(res_state, op_state, lang_summary_rep)
        self.attentions["final_pool_attn_wt"] = pool_attn_wt
        
        text_attentions = torch.stack(text_attentions)
        self.attentions["text"] = text_attentions
        spatial_attentions = torch.stack(spatial_attentions)
        self.attentions["image"] = spatial_attentions
        if(inter_mem_op_atts is not None and len(inter_mem_op_atts)>0):
            if(self.memory_window_len>0): #do padding before stacking
                max_k_len = max([att_x.size(2) for att_x in inter_mem_op_atts]) #BxQxKx1
                new_list = []
                for att_x in inter_mem_op_atts:
                    b, q, k, d = att_x.shape
                    #print(att_x.shape)
                    if(k<max_k_len):
                        new_list.append(torch.cat([att_x, torch.zeros(b, q, max_k_len-k, 1).to(att_x)], dim= 2))
                    else:
                        new_list.append(att_x)
                inter_mem_op_atts = new_list        
            
            self.attentions["inter_mem_op"] = torch.stack(inter_mem_op_atts)

        return final_res_state #, (op_states, res_states)
