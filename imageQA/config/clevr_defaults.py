from yacs.config import CfgNode as CN

_C = CN()

#_C.MODEL_TYPE = 'imanet'
_C.CLEVR_TRAIN_SPLIT = 'train' ##used in case want to train on train_original or trainA=trainB and test on humans/cogen/etc
_C.CLEVR_VAL_SPLIT = 'val'
_C.USE_CLEVR_HUMANS = 'original_clevr_only' #original_clevr_but_humans_combined_vocab, humans_with_combined_vocab

_C.AUX_LAYER_FOR_MEMTHRESH = False

_C.CLEVR_DIRECTORY = "/data/user_data/sjaiswa3/iprm_datasets/clevr/CLEVR_v1.0/questions/" 
_C.CLEVR_HUMANS_DIRECTORY = "/data/user_data/sjaiswa3/iprm_datasets/clevr/CLEVR-Humans/"
_C.USE_VLM_DATALOADER = True

_C.MODELS_FOR_CLEVR_H_ONLY_DIRECTORY = None
# -----------------------------------------------------------------------------
# VLM
# -----------------------------------------------------------------------------
_C.VLM = CN()
_C.VLM.DIM = 512
_C.VLM.OUTPUT_CLASSES = 28
_C.VLM.INPUT_VIS_FEAT_RES = (16,16)
_C.VLM.IS_TRANSFORMER_VIS_FEAT = False #incase of aux class token
_C.VLM.MODULE_TYPE = 'iprm' #iprm, ca, perciever_ca, concat_att, sa_fusion, iprm_timespan_groups

_C.VLM.LANG_SUMMARY_REP_DROPOUT = 0.08
_C.VLM.RESET_EMBED_LAYER = False
_C.VLM.ADD_AUX_LANG_TOKEN = False
_C.VLM.AUX_LANG_SA_LAYER = False
_C.VLM.LANG_ENC_DROPOUT = 0.1
_C.VLM.LANG_SA_ACT_LAYER = 'elu'

_C.VLM.USE_PRETRAINED_LANG_ENCODER = False
_C.VLM.PRETRAINED_LANG_ENCODER_PATH = None
_C.VLM.REMOVE_FIRST_AND_LAST_SPECIAL_TOKENS_PRETRAINED_LANG_ENCODER = False
_C.VLM.PRETRAINED_LANG_EMBEDDING_PATH = "/data/user_data/sjaiswa3/iprm_ckpts/ckpts/ckp2/vqav2_embeddings.pth"
_C.VLM.PRETRAINED_LANG_TOKENIZER_PATH = '/data/user_data/sjaiswa3/iprm_ckpts/ckpts/ckp2/vqav2_tokenizer.pth'

_C.VLM.VIS_INPUT_FEATS = 1024
_C.VLM.VIS_STEM_DROPOUT = 0.18
_C.VLM.VIS_STEM_NONLIN = 'elu' #gelu, elu, relu
_C.VLM.STEM_FILTER_SIZE = 3
_C.VLM.STEM_PAD = 1
_C.VLM.ADD_VIS_POSITIONAL_EMBDS = False
_C.VLM.POS_DROPOUT = 0.1


_C.VLM.CLASSIFIER_DROPOUT = 0.15
_C.VLM.CLASSIFIER_NONLIN = 'elu'
_C.VLM.CLASSIFIER_TYPE = 'concat_lang_and_result'




# -----------------------------------------------------------------------------
# IPRM
# -----------------------------------------------------------------------------
_C.IPRM = CN()


_C.IPRM.NUM_MEMORY_TOKENS = 6
_C.IPRM.NUM_COMPUTATION_STEPS = 6
_C.IPRM.MEMORY_ATT_WINDOW_LEN = 0

_C.IPRM.TIME_SPAN_GROUPS = [[0,1,2,3], [4,5,6]]



_C.IPRM.LANG_BLOCK_NONLIN = 'tanh'
_C.IPRM.LANG_BLOCK_MLP_DROP = 0.1
_C.IPRM.LANG_BLOCK_ATTN_DROP = 0.0
_C.IPRM.LANG_BLOCK_USE_LN = False
_C.IPRM.ADD_AUX_LANG_TOKEN = False
_C.IPRM.DO_TIME_STEP_SPECIFIC_PROJ = False

_C.IPRM.VIS_RETRIEVAL_BLOCK_IMPL_TYPE = 'original_default'  #efficient_implement, op_summed_efficient_implement
_C.IPRM.VIS_BLOCK_DROP = 0.15
_C.IPRM.VIS_BLOCK_ATTN_DROP = 0.0
_C.IPRM.REDUCTION_R = 1
_C.IPRM.VIS_BLOCK_NONLIN = 'elu'
_C.IPRM.VIS_BLOCK_USE_LN = False
_C.IPRM.DO_CONCAT_PROJ_FOR_MOD_WTS = False

#_C.IPRM.USE_PREV_OP_STATE_AS_K_FOR_RESULT_ATT_IN_VIS_RETRIEVAL = True
#_C.IPRM.GATE_RESULT_IN_VIS_QUERY_FORMATION = True
#_C.IPRM.GATE_TYPE = 'feat_mod' #feat_mod, single_sigmoid, feat_mod_sigmoid

_C.IPRM.DO_MEMORY_UPDATE = True
_C.IPRM.DO_MEMORY_OP_INTERACTION = True
_C.IPRM.MEMORY_STATE_UPDATE_MASK_SELF_IN_ATTN = True
_C.IPRM.MEMORY_STATE_UPDATE_DO_KEY_PROJ_OF_LANG_STATE = False
_C.IPRM.MEMORY_STATE_UPDATE_RES_DROPOUT = 0.1
_C.IPRM.MEMORY_STATE_UPDATE_MOD_DROPOUT = 0.1
_C.IPRM.MEM_ATTN_MATRIX_DROP = 0.0

_C.IPRM.DO_TIME_SPECIFIC_PROJ = False
_C.IPRM.MEM_UPDATE_BLOCK_USE_LN = False
_C.IPRM.OP_INTERACTION_BLOCK_USE_LN = False
_C.IPRM.MEM_OP_UPDATE_NONLIN = 'identity'
_C.IPRM.MEM_RES_UPDATE_NONLIN = 'identity'


_C.IPRM.MEMORY_OP_STATE_INIT = 'random' #random, lang_projs
_C.IPRM.MEMORY_RES_STATE_INIT = 'random' #random, vis_projs_on_avp
_C.IPRM.MEMORY_RES_POOL_METHOD = 'lang_cuml_query_op' #random_token_query_pool_op, lang_cuml_query_op

##for IPRMv2

_C.IPRM.USE_PRIOR_RESULT_STATES_FOR_LANG_QUERY = False
_C.IPRM.PRIOR_RESULT_STATE_MLP_NONLIN_FOR_LANG_RETRIEVAL = 'tanh'

_C.IPRM.USE_SIMPLER_VIS_RETRIEVAL_BLOCK= False
_C.IPRM.USE_MLP_FOR_PRIOR_RESULTS_PROJ_IN_VISRETRIEVAL = False
_C.IPRM.USE_MLP_FOR_LANG_PROJ_IN_VISRETRIEVAL = False
_C.IPRM.NONLIN_FOR_LANG_PROJ_IN_VISRETRIEVAL = 'elu'

_C.IPRM.DO_CONCAT_PROJ_FOR_QUERY_LANG_RETRIEVAL = False
_C.IPRM.DO_CONCAT_PROJ_FOR_QUERY_VIS_RETRIEVAL = False

# -----------------------------------------------------------------------------
# Attention blocks
# -----------------------------------------------------------------------------
_C.CA = CN()
_C.CA.NUM_BLOCKS = 4
_C.CA.NUM_HEADS = 8
_C.CA.MLP_INTER_DIM = 512*4
_C.CA.DROP_ATT = 0.0
_C.CA.DROP_PROJ = 0.1
_C.CA.DROP_MLP = 0.1
_C.CA.MLP_ACT_LAYER = 'elu'
_C.CA.ADD_AUX_LANG_TOKEN = False
_C.CA.ADD_AUX_LANG_TOKEN_TO_FRONT = False

# -----------------------------------------------------------------------------
# MAC
# -----------------------------------------------------------------------------
_C.MAC = CN()
_C.MAC.MAX_COMPUTATION_STEPS = 12
_C.MAC.USE_SELF_ATTN_MEMORY_UPDATE = True
_C.MAC.READ_DROPOUT = 0.15

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.DATASET_NAME = 'clevr'


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIM = 28


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.EVAL_BATCH_SIZE = 512
_C.DATALOADER.DATA_PERCENT = 1.0
_C.DATALOADER.FEATURES_PATH = "/data/user_data/sjaiswa3/iprm_datasets/clevr/clevr_feats/features_272x272"
_C.DATALOADER.TRAIN_SPLIT = "all"
_C.DATALOADER.VAL_SPLIT = "all"
_C.DATALOADER.LOAD_DECISION_MAKING_SUPERVISION = False
_C.DATALOADER.DECISION_MAKING_SUPERVISION_PATH = "./data_for_gating_supervision/CLEVR/all_separate_operations_and_classifiers/"
_C.DATALOADER.LOAD_CONJUNCT_FEATURE_AS_BINARY_NUMPY = False
_C.DATALOADER.IMAGES_PATH = None


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 35
_C.SOLVER.START_SCHEDULER_EPOCH = 0
_C.SOLVER.CLEVR_HUMAN_EPOCHS = 45
_C.SOLVER.LR = 1e-4
_C.SOLVER.LANG_ENC_LR = 1e-4 #1e-5
_C.SOLVER.USE_LANG_ENC_LR_FOR_FINETUNE = False

_C.SOLVER.GATE_LR = 0.0
_C.SOLVER.GRAD_CLIP = 8
_C.SOLVER.USE_SCHEDULER = True
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_INIT = 0.0
_C.SOLVER.FACTOR = 0.5
_C.SOLVER.PATIENCE = 0
_C.SOLVER.CLEVR_H_PATIENCE = 0
_C.SOLVER.THRESHOLD = 0.001





# ---------------------------------------------------------------------------- #
# weight saving/loading options
# ---------------------------------------------------------------------------- #
_C.SAVE_PATH = '/data/user_data/sjaiswa3/iprm_runs/clevr_runs'
_C.SAVE_DIRECTORY = '/data/user_data/sjaiswa3/iprm_runs/clevr_runs'
_C.SAVE_ATTENTIONS = False
_C.LOAD = False
_C.LOAD_PATH = ""
_C.DEVICE = "cuda"


# ---------------------------------------------------------------------------- #
# Wandb logging
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.EXPERIMENT_NAME = 'IPRM_Experiments'
_C.WANDB.PROJECT_NAME = 'iprm-experiments'

# ---------------------------------------------------------------------------- #
# Comet logging
# ---------------------------------------------------------------------------- #
_C.COMET = CN()
_C.COMET.EXPERIMENT_NAME = 'IPRM_Experiments'
_C.COMET.PROJECT_NAME = 'iprm-experiments'
