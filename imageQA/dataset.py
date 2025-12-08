import os
import pickle
import numpy as np
from PIL import Image
import h5py
import json
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from transforms import Scale
from vlm_data_utils import CLEVR_for_VLM_dataloader, collate_fn_vlm

def get_dataloader(cfg, mode="clevr", use_clevr_humans= False, 
                   clevr_train_split='train', #used in case want to train on train_original and test on humans/cogen/etc
                   clevr_val_split='val',
                   train_percent=1.0,
                   pin_memory=True,
                   #is_clevr_closure=False,
                   #load_decision_making_supervision=True
                  ):
    img_feats_path = os.path.join(cfg.DATALOADER.FEATURES_PATH)#, "features")

    train_clevr = CLEVR_for_VLM_dataloader(root = cfg.CLEVR_DIRECTORY if not use_clevr_humans else cfg.CLEVR_HUMANS_DIRECTORY,#cfg.ROOT_DATASET_DIRECTORY, 
         split = clevr_train_split,#='train', 
         model_config=None,
         train_percent = train_percent,#cfg.DATALOADER.TRAIN_PERCENT,
         images_root_path = cfg.DATALOADER.IMAGES_PATH, #os.path.join(, " #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
         load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
         img_feats_path=os.path.join(img_feats_path, "train_features.hdf5"),
         make_new_label_config=True,
         is_clevr_humans = use_clevr_humans,
                                #is_clevr_closure=is_clevr_closure
                               )
    if(train_percent<0.5):
        tr_drop_last=False
    else:
        tr_drop_last=True
    train_dataloader = DataLoader(
                        train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=tr_drop_last, shuffle=True, pin_memory=pin_memory,)
    val_clevr = CLEVR_for_VLM_dataloader(root = cfg.CLEVR_DIRECTORY if not use_clevr_humans else cfg.CLEVR_HUMANS_DIRECTORY,
                 split = clevr_val_split,#='train', 
                 model_config=None,
                 images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                 load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                 img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                 make_new_label_config=True,
                 is_clevr_humans = use_clevr_humans,
                  #clevr_closure_question_type = clevr_closure_question_type
                 #clevr_closure_set_type=clevr_closure_set_type
                             )
    
    val_dataloader = DataLoader(val_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        
    return train_dataloader, [val_dataloader]

def get_dataloader_for_closure(cfg, 
                   #clevr_train_split='train', #used in case want to train on train_original and test on humans/cogen/etc
                   #clevr_val_split='val',
                   train_percent=1.0,
                   pin_memory=True,
                   #is_clevr_closure=False,
                   #load_decision_making_supervision=True
                  ):
    img_feats_path = os.path.join(cfg.DATALOADER.FEATURES_PATH)#, "features")
    
    train_clevr = CLEVR_for_VLM(root = './Closure/',#cfg.ROOT_DATASET_DIRECTORY, 
         split = 'train',#='train', 
         model_config=None,
         train_percent = train_percent,#cfg.DATALOADER.TRAIN_PERCENT,
         images_root_path = cfg.DATALOADER.IMAGES_PATH, #os.path.join(, " #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
         load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
         img_feats_path=os.path.join(img_feats_path, "train_features.hdf5"),
         make_new_label_config=True,
         is_clevr_humans = False,
        clevr_closure_question_type = 'train_collated252',
                                #is_clevr_closure=is_clevr_closure
                               )
    if(train_percent<0.5):
        tr_drop_last=False
    else:
        tr_drop_last=True
    train_dataloader = DataLoader(
                        train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=tr_drop_last, shuffle=True, pin_memory=pin_memory,)
    
    baseline_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    for q_type in ['and_mat_spa', 'compare_mat', 'compare_mat_spa', 'embed_mat_spa', 'embed_spa_mat', 'or_mat_spa', 'or_mat']:
        val_clevr = CLEVR_for_VLM(root = './Closure/',
                     split = 'val',#='train', 
                     model_config=None,
                     images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                     load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                     img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                     make_new_label_config=True,
                     is_clevr_humans = False,
                      clevr_closure_question_type = q_type,
                      
                      #clevr_closure_question_type = clevr_closure_question_type
                     #clevr_closure_set_type=clevr_closure_set_type
                                 )
        val_dataloader = DataLoader(val_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        val_dataloaders.append(val_dataloader)
        
        
        test_clevr = CLEVR_for_VLM(root = './Closure/',
                     split = 'test',#='train', 
                     model_config=None,
                     images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                     load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                     img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                     make_new_label_config=True,
                     is_clevr_humans = False,
                      clevr_closure_question_type = q_type,
                      
                      #clevr_closure_question_type = clevr_closure_question_type
                     #clevr_closure_set_type=clevr_closure_set_type
                                 )
        test_dataloader = DataLoader(test_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        test_dataloaders.append(test_dataloader)
        
        baseline_clevr = CLEVR_for_VLM(root = './Closure/',
                     split = 'baseline',#='train', 
                     model_config=None,
                     images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                     load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                     img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                     make_new_label_config=True,
                     is_clevr_humans = False,
                      clevr_closure_question_type = q_type,
                      
                      #clevr_closure_question_type = clevr_closure_question_type
                     #clevr_closure_set_type=clevr_closure_set_type
                                 )
        baseline_dataloader = DataLoader(baseline_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        baseline_dataloaders.append(baseline_dataloader)
        
        
        

        
        
    return train_dataloader, val_dataloaders, test_dataloaders, baseline_dataloaders

def get_dataloader_for_clevroriginaltrain_and_closure(cfg, 
                   #clevr_train_split='train', #used in case want to train on train_original and test on humans/cogen/etc
                   #clevr_val_split='val',
                   train_percent=1.0,
                   pin_memory=True,
                   #is_clevr_closure=False,
                   #load_decision_making_supervision=True
                  ):
    img_feats_path = os.path.join(cfg.DATALOADER.FEATURES_PATH)#, "features")
    
    train_clevr = CLEVR_for_VLM(root = cfg.CLEVR_DIRECTORY,#cfg.ROOT_DATASET_DIRECTORY, 
         split ='train', 
         model_config=None,
         train_percent = train_percent,#cfg.DATALOADER.TRAIN_PERCENT,
         images_root_path = cfg.DATALOADER.IMAGES_PATH, #os.path.join(, " #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
         load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
         img_feats_path=os.path.join(img_feats_path, "train_features.hdf5"),
         make_new_label_config=True,
         is_clevr_humans = False,
                                #is_clevr_closure=is_clevr_closure
                               )
    if(train_percent<0.5):
        tr_drop_last=False
    else:
        tr_drop_last=True
    train_dataloader = DataLoader(
                        train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=tr_drop_last, shuffle=True, pin_memory=pin_memory,)
    
    baseline_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    for q_type in ['and_mat_spa', 'compare_mat', 'compare_mat_spa', 'embed_mat_spa', 'embed_spa_mat', 'or_mat_spa', 'or_mat']:
        val_clevr = CLEVR_for_VLM(root = './Closure/',
                     split = 'val',#='train', 
                     model_config=None,
                     images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                     load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                     img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                     make_new_label_config=True,
                     is_clevr_humans = False,
                      clevr_closure_question_type = q_type,
                      
                      #clevr_closure_question_type = clevr_closure_question_type
                     #clevr_closure_set_type=clevr_closure_set_type
                                 )
        val_dataloader = DataLoader(val_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        val_dataloaders.append(val_dataloader)
        
        
        test_clevr = CLEVR_for_VLM(root = './Closure/',
                     split = 'test',#='train', 
                     model_config=None,
                     images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                     load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                     img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                     make_new_label_config=True,
                     is_clevr_humans = False,
                      clevr_closure_question_type = q_type,
                      
                      #clevr_closure_question_type = clevr_closure_question_type
                     #clevr_closure_set_type=clevr_closure_set_type
                                 )
        test_dataloader = DataLoader(test_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        test_dataloaders.append(test_dataloader)
        
        baseline_clevr = CLEVR_for_VLM(root = './Closure/',
                     split = 'baseline',#='train', 
                     model_config=None,
                     images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                     load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                     img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                     make_new_label_config=True,
                     is_clevr_humans = False,
                      clevr_closure_question_type = q_type,
                      
                      #clevr_closure_question_type = clevr_closure_question_type
                     #clevr_closure_set_type=clevr_closure_set_type
                                 )
        baseline_dataloader = DataLoader(baseline_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, pin_memory=pin_memory,)
        baseline_dataloaders.append(baseline_dataloader)
        
        
        

        
        
    return train_dataloader, val_dataloaders, test_dataloaders, baseline_dataloaders



    
    """
    else:
            train_clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, train_percent=cfg.DATALOADER.DATA_PERCENT, use_clevr_humans= use_clevr_humans,
                            split=clevr_train_split, cfg=cfg
                           )
            train_dataloader = DataLoader(
                train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, shuffle=True
            )
        
        if(use_vlm_dataloader):
            val_clevr = CLEVR_for_VLM(root = cfg.CLEVR_DIRECTORY if not use_clevr_humans else cfg.CLEVR_HUMANS_DIRECTORY,
                 split = clevr_val_split,#='train', 
                 model_config=None,
                 images_root_path = cfg.DATALOADER.IMAGES_PATH, #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                 load_img_feats=True if cfg.DATALOADER.FEATURES_PATH!=None else False,
                 img_feats_path=os.path.join(img_feats_path, "val_features.hdf5"),
                 make_new_label_config=True,
                 is_clevr_humans = use_clevr_humans)
            val_dataloader = DataLoader(
                val_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn_vlm, drop_last=False, 
            )
        else:
            val_clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, split=clevr_val_split, use_clevr_humans= use_clevr_humans, cfg=cfg)
        
            val_dataloader = DataLoader(
                val_clevr, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=False, 
            )
        return train_dataloader, [val_dataloader]
    elif mode == "gqa":
        train_gqa = GQA(cfg.DATALOADER.FEATURES_PATH, 'train', cfg.DATALOADER.TRAIN_SPLIT, train_percent=cfg.DATALOADER.DATA_PERCENT, cfg=cfg,)
        train_dataloader = DataLoader(
            train_gqa, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data_GQA, drop_last=True, shuffle=True,
        )
        val_gqa = GQA(cfg.DATALOADER.FEATURES_PATH, 'testdev', cfg.DATALOADER.VAL_SPLIT, cfg=cfg)
        val_dataloader = DataLoader(
            val_gqa, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data_GQA, drop_last=True
        )
        return train_dataloader, [val_dataloader]
    elif(mode == "clevr-cogen"):
        train_clevr = CLEVR_Cogen(cfg.DATALOADER.FEATURES_PATH, train_percent=cfg.DATALOADER.DATA_PERCENT,
                            split=clevr_train_split, cfg=cfg
                           )
        train_dataloader = DataLoader(
            train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, shuffle=True
        )
        val_clevr1 = CLEVR_Cogen(cfg.DATALOADER.FEATURES_PATH, split='valA', cfg=cfg)
        val_dataloader1 = DataLoader(
            val_clevr1, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, 
        )
        val_clevr2 = CLEVR_Cogen(cfg.DATALOADER.FEATURES_PATH, split='valB', cfg=cfg)
        val_dataloader2 = DataLoader(
            val_clevr2, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, 
        )
        return train_dataloader, [val_dataloader1, val_dataloader2]
    elif(mode == "clevr-cogen-finetune"):
        train_clevr = CLEVR_Cogen(cfg.DATALOADER.FEATURES_PATH, train_percent=cfg.DATALOADER.DATA_PERCENT,
                            split='finetunetrainB', cfg=cfg
                           )
        train_dataloader = DataLoader(
            train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, shuffle=True
        )
        val_clevr1 = CLEVR_Cogen(cfg.DATALOADER.FEATURES_PATH, split='valA', cfg=cfg)
        val_dataloader1 = DataLoader(
            val_clevr1, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, 
        )
        val_clevr2 = CLEVR_Cogen(cfg.DATALOADER.FEATURES_PATH, split='finetunevalB', cfg=cfg)
        val_dataloader2 = DataLoader(
            val_clevr2, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, 
        )
        return train_dataloader, [val_dataloader1, val_dataloader2]
    
    elif mode == "clevr_original_and_humans_mix":
        train_clevr1 = CLEVR(cfg.DATALOADER.FEATURES_PATH, train_percent=cfg.DATALOADER.DATA_PERCENT, use_clevr_humans= use_clevr_humans,
                            split=clevr_train_split, cfg=cfg
                           )
        #train_dataloader1 = DataLoader(
            #train_clevr1, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, shuffle=True
        #)
        train_clevr2 = CLEVR(cfg.DATALOADER.FEATURES_PATH, train_percent=cfg.DATALOADER.DATA_PERCENT, use_clevr_humans= use_clevr_humans,
                            split=clevr_train_split, cfg=cfg
                           )
        combined_clevr = ConcatDataset([train_clevr1, train_clevr2])
        
        train_dataloader_concat = DataLoader(
            combined_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, shuffle=True
        )
        val_clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, split=clevr_val_split, use_clevr_humans= use_clevr_humans)
        val_dataloader = DataLoader(
            val_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, 
        )
        return train_dataloader_concat, [val_dataloader]
    else:
        raise NotImplementedError()
    return train_dataloader, [val_dataloader]
    """
    
def filter_operations(original_data, operations_to_filter):
        new_set_without_ops = []
        x = original_data
        for el in x:
            program = el[-1]
            skip=False
            for step in program[1:]:
                if(step['function'] in operations_to_filter): #['scene','intersect','union']):
                    skip=True
                    break
                for op in operations_to_filter:
                    if(step['function'].startswith(op)):
                        skip=True
                        break
            if(not skip):
                new_set_without_ops.append(el)
        print("Filtered set len: {} . original set len was: {}".format(len(new_set_without_ops), len(x)))
        
              
        return new_set_without_ops
    
class CLEVR(Dataset):
    def __init__(self, root, split='train', train_percent=1.0, use_clevr_humans='original_clevr_only', cfg=None
                ):
        features_path = os.path.join(root, 'features')
        with open('{}/clevr_original_{}.pkl'.format(features_path, split), 'rb') as f:
            data = pickle.load(f)
        if(False and len(cfg.IMANet.FILTER_OPERATIONS)>0):
            print("**Loading FILTERED set of data with {} operations not included".format(cfg.IMANet.FILTER_OPERATIONS))
            self.data = filter_operations(data, cfg.IMANet.FILTER_OPERATIONS)
        else:
            self.data = data
        self.load_decision_making_supervision = cfg.DATALOADER.LOAD_DECISION_MAKING_SUPERVISION
        self.end_of_compute_operation_token = 0
        self.end_of_compute_feat_value_token = 0
        if(cfg is not None and self.load_decision_making_supervision):
            with open(os.path.join(cfg.DATALOADER.DECISION_MAKING_SUPERVISION_PATH, "classifiers_key2id.json"),'rb') as f:
                self.classifiers_mapping = json.load(f)
            with open(os.path.join(cfg.DATALOADER.DECISION_MAKING_SUPERVISION_PATH, "feature_values_key2id.json"), 'rb') as f:
                self.feat_values_mapping = json.load(f)
            with open(os.path.join(cfg.DATALOADER.DECISION_MAKING_SUPERVISION_PATH, "operations_key2id.json"),'rb') as f:
                self.operations_mapping = json.load(f)
                
            self.end_of_compute_operation_token = self.operations_mapping['already_ended_compute']
            self.end_of_compute_feat_value_token = self.feat_values_mapping['already_ended_compute']
        
        self.root = root
        self.split = split
        self.train_percent = train_percent
        self.cfg = cfg
        self.h = h5py.File('{}/{}_features.hdf5'.format(features_path, split), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()
        
    
    
    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[int(index/self.train_percent)]#, program = self.data[int(index/self.train_percent)]
        #print('ch1', question)
        program=None
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        if(False and self.load_decision_making_supervision):
            ops_seq, feat_values_seq, classifier_type = map_program_operation_and_value_decision_sequences(program, 
                                                   operations_dict= self.operations_mapping,
                                                   classifier_dict= self.classifiers_mapping,
                                                   feature_values_dict= self.feat_values_mapping,
                                                   max_conjunct_len=self.cfg.IMANet.MAX_FEAT_CONJUNCT_LEN,
                                                   conjunctive_types=self.cfg.IMANet.FEAT_CONJUNCTIVE_TYPES,
                                                    skip_first_step=True, #skips scene
                                                    store_activated_values_as_binary_features=self.cfg.DATALOADER.LOAD_CONJUNCT_FEATURE_AS_BINARY_NUMPY, #will store as numpy array if True
                                                    count_spatial_relation_as_feature_search_relation = self.cfg.IMANet.COUNT_SPATIAL_RELATION_AS_FEATURE_SEARCH_RELATION                                                 
                                                  )
        else:
            ops_seq, feat_values_seq, classifier_type = (None, None, None)
        return img, question, len(question), answer, family, index, program, ops_seq, feat_values_seq, classifier_type, self.end_of_compute_operation_token, self.end_of_compute_feat_value_token, self.cfg.DATALOADER.LOAD_CONJUNCT_FEATURE_AS_BINARY_NUMPY

    def __len__(self):
        return int(len(self.data) * self.train_percent)

class CLEVR_Cogen(Dataset):
    def __init__(self, root, split='trainA', train_percent=1.0):
        features_path = os.path.join(root, 'features')
        with open('{}/clevrcogen_original_{}.pkl'.format(features_path, split), 'rb') as f:
            data = pickle.load(f)
            #with open('{}/clevr_humans_and_original_{}.pkl'.format(features_path, split), 'rb') as f:
             #   self.data = pickle.load(f)
        if(False and cfg.IMANet.FILTER_OPERATIONS is not None):
            print("**Loading FILTERED set of data with {} operations not included".format(cfg.IMANet.FILTER_OPERATIONS))
            self.data = filter_operations(data, cfg.IMANet.FILTER_OPERATIONS)
        else:
            self.data = data
        self.load_decision_making_supervision = cfg.DATALOADER.LOAD_DECISION_MAKING_SUPERVISION
        if(cfg is not None and self.load_decision_making_supervision):
            with open(os.path.join(cfg.DATALOADER.DECISION_MAKING_SUPERVISION_PATH, "classifiers_key2id.json"),'rb') as f:
                self.classifiers_mapping = json.load(f)
            with open(os.path.join(cfg.DATALOADER.DECISION_MAKING_SUPERVISION_PATH, "feature_values_key2id.json"), 'rb') as f:
                self.feat_values_mapping = json.load(f)
            with open(os.path.join(cfg.DATALOADER.DECISION_MAKING_SUPERVISION_PATH, "operations_key2id.json"),'rb') as f:
                self.operations_mapping = json.load(f)
                
        self.root = root
        self.split = split
        self.train_percent = train_percent
        
        self.h = h5py.File('{}/{}_features.hdf5'.format(features_path, split), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()
        
    def _filter_operations(self, original_data, operations_to_filter):
        new_set_without_ops = []
        x = original_data
        for el in x:
            program = el[-1]
            skip=False
            for step in program[1:]:
                if(step['function'] in operations_to_filter): #['scene','intersect','union']):
                    skip=True
                    break
            if(not skip):
                new_set_without_ops.append(el)
        return new_set_without_ops
    
    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[int(index/self.train_percent)]

        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])
        
        if(False and self.load_decision_making_supervision):
            ops_seq, feat_values_seq, classifier_type = map_program_operation_and_value_decision_sequences(program, 
                                                   operations_dict= self.operations_mapping,
                                                   classifier_dict= self.classifiers_mapping,
                                                   feature_values_dict= self.feat_values_mapping,
                                                   max_conjunct_len=self.cfg.IMANet.MAX_FEAT_CONJUNCT_LEN,
                                                   conjunctive_types=self.cfg.IMANet.FEAT_CONJUNCTIVE_TYPES,
                                                    skip_first_step=True, #skips scene
                                                    store_activated_values_as_binary_features=True, #will store as numpy array
                                                  )
        else:
            ops_seq, feat_values_seq, classifier_type = (None, None, None)

        return img, question, len(question), answer, family, index, program, ops_seq, feat_values_seq, classifier_type

    def __len__(self):
        return int(len(self.data) * self.train_percent)
    