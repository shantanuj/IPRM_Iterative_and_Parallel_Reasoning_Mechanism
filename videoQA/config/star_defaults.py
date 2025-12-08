from yacs.config import CfgNode as CN

_C = CN()

#_C.MODEL_TYPE = 'imanet'
_C.TRAIN_SPLIT = 'train' ##used in case want to train on train_original or trainA=trainB and test on humans/cogen/etc
_C.VAL_SPLIT = 'val'


_C.TRAIN_DATASET_DIRECTORY = "/data/user_data/sjaiswa3/iprm_datasets/star_data/STAR_train.json"
_C.TEST_DATASET_DIRECTORY = "/data/user_data/sjaiswa3/iprm_datasets/star_data/STAR_val.json"
_C.FINAL_TEST_DATASET_DIRECTORY = "/data/user_data/sjaiswa3/iprm_datasets/star_data/STAR_test.json"

# -----------------------------------------------------------------------------
# VLM
# -----------------------------------------------------------------------------
_C.VLM = CN()
_C.VLM.VLM_LANG_ONLY_BASELINE = False

_C.VLM.DIM = 512
_C.VLM.OUTPUT_CLASSES = 4
_C.VLM.PROCESS_EACH_OPTION_INDEPENDENTLY = True
_C.VLM.INPUT_VIS_FEAT_RES = (14,14)
_C.VLM.IS_TRANSFORMER_VIS_FEAT = False #incase of aux class token
_C.VLM.MODULE_TYPE = 'iprm' #rmam, ca, perciever_ca, concat_att, sa_fusion, rmam_timespan_groups

_C.VLM.LANG_SUMMARY_REP_DROPOUT = 0.08
_C.VLM.RESET_EMBED_LAYER = False
_C.VLM.ADD_AUX_LANG_TOKEN = True
_C.VLM.AUX_LANG_SA_LAYER = False
_C.VLM.LANG_ENC_DROPOUT = 0.1
_C.VLM.LANG_SA_ACT_LAYER = 'elu'

_C.VLM.USE_PRETRAINED_LANG_ENCODER = True #False
_C.VLM.PRETRAINED_LANG_ENCODER_PATH = ''
_C.VLM.PRETRAINED_LANG_TOKENIZER_PATH = '/data/user_data/sjaiswa3/iprm_datasets/BLIPv1_model_components/vqav2_tokenizer_eval.pth'
_C.VLM.REMOVE_FIRST_AND_LAST_SPECIAL_TOKENS_PRETRAINED_LANG_ENCODER = True
_C.VLM.PRETRAINED_LANG_EMBEDDING_PATH = ""
#_C.VLM.PRETRAINED_LANG_TOKENIZER_PATH = '/data/usrdata/shan/BLIPv1_model_components/vqav2_tokenizer.pth'

_C.VLM.LSTM_APPLY_AUX_PROJS = False
_C.VLM.SHARE_QUESTION_AND_OPTION_ENCODER = True
_C.VLM.SHARE_QUESTION_AND_OPTION_PROJS_METHOD = 'share_options' #share_options, share_question_and_options
_C.VLM.ADD_LANG_AND_CHOICE_POSITIONAL_EMBEDS = False
_C.VLM.COMBINE_QUES_AND_OPTION_SUMMARY_FOR_MODULE = True
_C.VLM.CONCAT_QUESTION_AND_OPTION_LANG_REPS_FOR_MODULE = True


_C.VLM.VIS_INPUT_FEATS = 1024
_C.VLM.VIS_STEM_DROPOUT = 0.18
_C.VLM.VIS_STEM_NONLIN = 'elu' #gelu, elu, relu
_C.VLM.STEM_FILTER_SIZE = 3
_C.VLM.STEM_PAD = 1
_C.VLM.ADD_VIS_POSITIONAL_EMBDS = False
_C.VLM.POS_DROPOUT = 0.1

_C.VLM.USE_FRAME_POS_EMBEDS = True


_C.VLM.NORMALIZE_BBOX_AND_PERSON_COORDS = True

_C.VLM.PERSON_BBOX_LATENT_DIMS = 256
_C.VLM.PERSON_BBOX_LATENT_DROP = 0.2

_C.VLM.USE_PERSON_EMBEDDING = False
_C.VLM.UNIFORM_INIT_PERSON_EMBD = True
_C.VLM.PERSON_EMBD_DIMS = 256
_C.VLM.PERSON_EMBD_DROPOUT = 0.15
_C.VLM.PERSON_EMBD_LATENT_DIMS = 128
_C.VLM.PERSON_EMBED_LATENT_DROP = 0.15

_C.VLM.NUM_PERSON_KEYPOINTS = 17
_C.VLM.PERSON_COORDS_LATENT_METHOD = 'indv_projs_then_sum' #indv_projs_then_sum, concat_coords_then_proj
_C.VLM.PERSON_COORDS_LATENT_DIMS = 256
_C.VLM.PERSON_COORDS_LATENT_DROP = 0.2
_C.VLM.CONCAT_PERSON_VIS_DROPOUT = 0.0

_C.VLM.OBJ_BBOX_LATENT_DIMS = 128
_C.VLM.OBJ_BBOX_LATENT_DROP = 0.15

_C.VLM.OBJ_LABEL_EMBD_LAYER_PATH = "./glove_obj_label_embeddings.pth"
_C.VLM.RESET_OBJ_LABEL_EMBED_LAYER = False
_C.VLM.OBJ_LABEL_EMBD_DIMS = 300
_C.VLM.OBJ_LABEL_EMBD_DROPOUT = 0.15
_C.VLM.OBJ_LABEL_LATENT_DIMS = 128
_C.VLM.OBJ_LABEL_LATENT_DROP = 0.15

_C.VLM.USE_COMBINED_OBJ_REL_EMBED = False
_C.VLM.NUM_OBJ_RELS = 28
_C.VLM.NUM_OBJ_ATT_RELS = 5
_C.VLM.NUM_OBJ_SPATIAL_RELS = 8 
_C.VLM.NUM_OBJ_CONTACT_RELS = 19

_C.VLM.OBJ_REL_EMBD_DIMS = 256
_C.VLM.OBJ_REL_EMBD_DROPOUT = 0.15
_C.VLM.OBJ_REL_LATENT_DIMS = 128
_C.VLM.OBJ_REL_LATENT_DROP = 0.15
_C.VLM.UNIFORM_INIT_OBJ_RELS_EMBED = True

_C.VLM.CONCAT_OBJ_VIS_DROPOUT = 0.0



_C.VLM.INDV_OPTION_CLASSIFIER_DROP = 0.08
_C.VLM.CLASSIFIER_DROPOUT = 0.15
_C.VLM.CLASSIFIER_NONLIN = 'elu'
_C.VLM.CLASSIFIER_TYPE = 'concat_lang_and_result'
_C.VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE = 'joint_concat'  #individual, joint_concat
_C.VLM.INDIVIDUAL_OPTION_CLASSIFIER_DIM = 128



# -----------------------------------------------------------------------------
# IPRM
# -----------------------------------------------------------------------------
_C.IPRM = CN()


_C.IPRM.NUM_MEMORY_TOKENS = 6
_C.IPRM.NUM_COMPUTATION_STEPS = 6
_C.IPRM.MEMORY_ATT_WINDOW_LEN = 0

_C.IPRM.TIME_SPAN_GROUPS = [[0,1,2,3, 4,5,6,7,8,9,10,11,12]]



_C.IPRM.LANG_BLOCK_NONLIN = 'tanh'
_C.IPRM.LANG_BLOCK_MLP_DROP = 0.1
_C.IPRM.LANG_BLOCK_ATTN_DROP = 0.0
_C.IPRM.LANG_BLOCK_USE_LN = False
_C.IPRM.ADD_AUX_LANG_TOKEN = False
_C.IPRM.DO_TIME_STEP_SPECIFIC_PROJ = False


_C.IPRM.VIS_BLOCK_DROP = 0.15
_C.IPRM.VIS_BLOCK_ATTN_DROP = 0.0
_C.IPRM.REDUCTION_R = 1
_C.IPRM.VIS_BLOCK_NONLIN = 'elu'
_C.IPRM.VIS_BLOCK_USE_LN = False

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

_C.IPRM.MEM_OP_UPDATE_NONLIN = 'identity'
_C.IPRM.MEM_RES_UPDATE_NONLIN = 'identity'

_C.IPRM.MEM_UPDATE_BLOCK_USE_LN = False
_C.IPRM.OP_INTERACTION_BLOCK_USE_LN = False



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
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.DATASET_NAME = 'star'


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIM = 4


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_SAMPLE_FRAMES = 32
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.EVAL_BATCH_SIZE = 128
_C.DATALOADER.TRAIN_PERCENT = 1.0
_C.DATALOADER.VIDEOS_PATH = "/data/user_data/sjaiswa3/iprm_datasets/charades_data/Charades_v1_480/"
_C.DATALOADER.OBJ_PROPOSALS_PATH = "/data/user_data/sjaiswa3/iprm_datasets/star_data/obj_bboxes_by_videoid.pkl"
_C.DATALOADER.PERSON_PROPOSALS_PATH = "/data/user_data/sjaiswa3/iprm_datasets/star_data/person_bboxes_by_videoid.pkl"
_C.DATALOADER.DO_BATCH_WISE_MAX_OBJS = False

_C.DATALOADER.MIN_REL_SCORE_THRESH = -1000.0
_C.DATALOADER.MIN_OBJ_SCORE_THRESH = -1000.0
_C.DATALOADER.DO_MUTUALLY_EXCL_REL_FILTERING = False
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 35
_C.SOLVER.START_SCHEDULER_EPOCH = 0
_C.SOLVER.LR = 1e-4
_C.SOLVER.LANG_ENC_LR = 1e-4 #1e-5
_C.SOLVER.USE_LANG_ENC_LR_FOR_FINETUNE = False

_C.SOLVER.CRITERION = 'binary_cross_entropy' #binary_cross_entropy, cross_entropy
_C.SOLVER.GRAD_CLIP = 8
_C.SOLVER.USE_SCHEDULER = True
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_INIT = 0.0
_C.SOLVER.FACTOR = 0.5
_C.SOLVER.PATIENCE = 0
_C.SOLVER.THRESHOLD = 0.001





# ---------------------------------------------------------------------------- #
# weight saving/loading options
# ---------------------------------------------------------------------------- #
_C.SAVE_DIRECTORY = '/data/usrdata/shan/STAR_VLMs'
_C.SAVE_ATTENTIONS = False
_C.LOAD = False
_C.LOAD_PATH = ""
_C.DEVICE = "cuda"


# ---------------------------------------------------------------------------- #
# Wandb logging
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.EXPERIMENT_NAME = 'STAR experiments'
_C.WANDB.PROJECT_NAME = 'iprm-star-experiments'
