import os
import pickle
import numpy as np
from PIL import Image
import h5py
import json
from glob import glob
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from data_utils import *

def get_dataloader(cfg, 
                   train_percent=1.0,
                   ):
    train_ds = STAR_for_VLM(root = cfg.TRAIN_DATASET_DIRECTORY,#, f"STAR_{train_split}.json"), #cfg.ROOT_DATASET_DIRECTORY, 
         num_sample_frames = cfg.DATALOADER.NUM_SAMPLE_FRAMES,
         videos_root_path = cfg.DATALOADER.VIDEOS_PATH, #os.path.join(, " #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
         obj_proposals_path = cfg.DATALOADER.OBJ_PROPOSALS_PATH, 
         person_proposals_path = cfg.DATALOADER.PERSON_PROPOSALS_PATH,
        #make_new_label_config=True,
         train_percent = train_percent,#cfg.DATALOADER.TRAIN_PERCENT,
         #num_expected_sampled_frames = cfg.VLM.EXPECTED_NUM_FRAMES,
         #load_segmask=cfg.VLM.INTEGRATE_SEGMASK_INFO
         min_rel_score_thresh=cfg.DATALOADER.MIN_REL_SCORE_THRESH,
         min_obj_score_thresh=cfg.DATALOADER.MIN_OBJ_SCORE_THRESH,
        do_mutually_exclusive_rel_filtering=cfg.DATALOADER.DO_MUTUALLY_EXCL_REL_FILTERING,
       )
    if(cfg.DATALOADER.DO_BATCH_WISE_MAX_OBJS):
        collate_fn = collate_fn_batch_wise_max_objs
    else:
        collate_fn = collate_fn_star
    train_dataloader = DataLoader(
                        train_ds, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn, drop_last=True, shuffle=True)
    
    val_ds = STAR_for_VLM(root = cfg.TEST_DATASET_DIRECTORY,
                 num_sample_frames = cfg.DATALOADER.NUM_SAMPLE_FRAMES,
                 videos_root_path = cfg.DATALOADER.VIDEOS_PATH, #os.path.join(, " #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                 obj_proposals_path = cfg.DATALOADER.OBJ_PROPOSALS_PATH, 
                 person_proposals_path = cfg.DATALOADER.PERSON_PROPOSALS_PATH,
                  min_rel_score_thresh=cfg.DATALOADER.MIN_REL_SCORE_THRESH,
                 min_obj_score_thresh=cfg.DATALOADER.MIN_OBJ_SCORE_THRESH,
                  do_mutually_exclusive_rel_filtering=cfg.DATALOADER.DO_MUTUALLY_EXCL_REL_FILTERING,
                 )
    val_dataloader = DataLoader(val_ds, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn, drop_last=False)
        
    return train_dataloader, [val_dataloader]

def get_test_dataloader(cfg, 
                   ):
    test_ds = STAR_for_VLM(root = cfg.FINAL_TEST_DATASET_DIRECTORY,
                 num_sample_frames = cfg.DATALOADER.NUM_SAMPLE_FRAMES,
                 videos_root_path = cfg.DATALOADER.VIDEOS_PATH, #os.path.join(, " #"/data/dataset/CLEVR/CLEVR_v1.0/images/",
                 obj_proposals_path = cfg.DATALOADER.OBJ_PROPOSALS_PATH, 
                 person_proposals_path = cfg.DATALOADER.PERSON_PROPOSALS_PATH,
                  min_rel_score_thresh=cfg.DATALOADER.MIN_REL_SCORE_THRESH,
                 min_obj_score_thresh=cfg.DATALOADER.MIN_OBJ_SCORE_THRESH,
                   do_mutually_exclusive_rel_filtering=cfg.DATALOADER.DO_MUTUALLY_EXCL_REL_FILTERING,
                   without_labels=True
                 )
    if(cfg.DATALOADER.DO_BATCH_WISE_MAX_OBJS):
        collate_fn = collate_fn_batch_wise_max_objs
    else:
        collate_fn = collate_fn_star
    test_dataloader = DataLoader(test_ds, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn, drop_last=False)
        
    return test_dataloader


class STAR_for_VLM(torch.utils.data.Dataset):
    def __init__(self, 
                 root, 
                 num_sample_frames = 44,
                 videos_root_path = "/data/dataset/charades/video/Charades_v1_480/",
                 #video_proposals_path = "/data/dataset/CLEVRER/proposals/",
                 obj_proposals_path = "/data/usrdata/shan/STAR_VideoQA/obj_bboxes_by_videoid.pkl",
                 person_proposals_path = "/data/usrdata/shan/STAR_VideoQA/person_bboxes_by_videoid.pkl",
                 #answer_vocab_path = './clevrer_labels_tuple.pkl',
                 obj_label_vocab_path = './star_obj_labels_tuple.pkl',
                 relation_vocab_path_root  = "./", # "./star_rel_labels.pkl",
                 min_rel_score_thresh=-1000,
                 min_obj_score_thresh=-1000,
                 do_mutually_exclusive_rel_filtering=False,
                 train_percent=1.0, 
                 print_warnings=False,
                 without_labels=False,
                 question_to_video_frames_path = "./star_keyframes_by_question_dict.pkl",
                ):
        self.data = json.load(open(root,'rb'))
        
        self.min_obj_score_thresh = min_obj_score_thresh
        self.min_rel_score_thresh = min_rel_score_thresh
        self.videos_root_path = videos_root_path
        self.obj_proposals_path = obj_proposals_path
        self.person_proposals_path = person_proposals_path
        self.obj_proposals_struct = pickle.load(open(obj_proposals_path,'rb'))
        self.person_proposals_struct = pickle.load(open(person_proposals_path,'rb'))
        self.do_mutually_exclusive_rel_filtering = do_mutually_exclusive_rel_filtering
        if(min_rel_score_thresh > -1000  or min_obj_score_thresh>-1000):
            print(f"Performing filtering of obj and person proposals with min_rel_score_thresh {min_rel_score_thresh} and min_obj_score_thresh {min_obj_score_thresh}")
            
            self.obj_proposals_struct, self.person_proposals_struct, max_objs_for_frame = self.filter_data_and_recompute_max_objs(self.obj_proposals_struct, self.person_proposals_struct, min_rel_score_thresh, min_obj_score_thresh)
        else:
            max_objs_for_frame = None
        if(self.do_mutually_exclusive_rel_filtering):
            print(f"Performing filtering of mutually exlcusive relations...")
            
            self.obj_proposals_struct = self.filter_mutually_exclusive_relations(self.obj_proposals_struct)
            
            
        self.num_sample_frames = num_sample_frames
        
        self.train_percent = train_percent 
        self.print_warnings=print_warnings
            
        if(not os.path.exists(obj_label_vocab_path)):
            print("WARNING: Creating new obj and rel label")
            self._create_obj_and_rel_label_vocab(obj_label_vocab_path, relation_vocab_path_root)
        
        self.obj_label2id, self.id2objlabel, self.objlabel_len, self.max_num_objs = pickle.load(open(obj_label_vocab_path,'rb'))
        if(max_objs_for_frame is not None):
            print(f"Max num objs per frame changed to {self.max_num_objs} post filtering")
            self.max_num_objs = max_objs_for_frame
        else:
            print(f"Max num objs per frame is {self.max_num_objs}")
        self.rel_label2id, self.id2rel_label, self.combined_rel_label_len = pickle.load(open(os.path.join(relation_vocab_path_root, "star_combined_rel_labels.pkl"),'rb'))
        self.att_rel_label2id, self.id2att_rel_label, self.att_rel_label_len = pickle.load(open(os.path.join(relation_vocab_path_root, "star_att_rel_labels.pkl"),'rb'))
        self.spatial_rel_label2id, self.id2spatial_rel_label, self.spatial_rel_label_len = pickle.load(open(os.path.join(relation_vocab_path_root, "star_spatial_rel_labels.pkl"),'rb'))
        self.contact_rel_label2id, self.id2contact_rel_label, self.contact_rel_label_len = pickle.load(open(os.path.join(relation_vocab_path_root, "star_contact_rel_labels.pkl"),'rb'))
        
        self.without_labels = without_labels
        self.q_to_vidid_and_frames_dict = pickle.load(open(question_to_video_frames_path, 'rb'))
        
    
    def check_all_in_items_exist(self, in_items, or_list):
        return all(item in or_list for item in in_items)

    def get_list_with_removed_elements(self,
                                       data_list,
                                  indices_to_remove_list
                                 ):
        return [element for index, element in enumerate(data_list) if index not in indices_to_remove_list]

    def filter_mutually_exclusive_relations(self, or_obj_proposals_struct):
                                       #or_person_proposals_struct, 
                                       #min_rel_score_thresh, 
                                       #min_obj_score_thresh):
        num_att_rels_removed_total = 0
        num_spatial_rels_removed_total = 0
        num_contact_rels_removed_total = 0
        
        num_orig_att_rels = 0
        num_orig_spatial_rels = 0
        num_orig_contact_rels = 0
        
        for video_id, video_struct in or_obj_proposals_struct.items():
            #print('hello')
            for frame_id, obj_struct_list in video_struct['frames'].items():
                obj_indices_to_remove = []
                new_obj_struct_list = []
                #print(len(obj_struct_list))
                #input()
                for obj_i, obj_struct in enumerate(obj_struct_list):
                    if('attention_relationship_scores' in obj_struct):
                        rel_indices_to_remove = set()
                        mut_excl_items_to_check=[['looking_at', 'not_looking_at']]
                        num_orig_att_rels+= len(obj_struct['attention_relationship_scores'])
                        for mut_exl_set in mut_excl_items_to_check:
                            #print(obj_struct['attention_relationship'], mut_exl_set)
                            #input()
                            if(self.check_all_in_items_exist(mut_exl_set, obj_struct['attention_relationship'])):
                                #print('hello')
                                max_rel_score_index = 0
                                max_rel_score=0
                                covered_indices = []
                                for rel_name in mut_exl_set:
                                    rel_index = obj_struct['attention_relationship'].index(rel_name)
                                    covered_indices.append(rel_index)
                                    if(obj_struct['attention_relationship_scores'][rel_index] > max_rel_score):
                                        max_rel_score_index = rel_index
                                        max_rel_score = obj_struct['attention_relationship_scores'][rel_index]
                                less_score_indices = set(covered_indices) - set([max_rel_score_index])
                                #print('aa',less_score_indices)
                                rel_indices_to_remove = rel_indices_to_remove.union(less_score_indices)
                                #print('bb', rel_indices_to_remove)
                                #input()
                                #for rel_n in mut_exl_set:
                        obj_struct['attention_relationship_scores'] = self.get_list_with_removed_elements(obj_struct['attention_relationship_scores'], list(rel_indices_to_remove))
                        obj_struct['attention_relationship'] = self.get_list_with_removed_elements(obj_struct['attention_relationship'], list(rel_indices_to_remove))
                        #print(rel_indices_to_remove)
                        #if(len(rel_indices_to_remove)>0):
                         #   print('cc', obj_struct['attention_relationship'])
                        num_att_rels_removed_total+=len(rel_indices_to_remove)

                    if('spatial_relationship_scores' in obj_struct):
                        rel_indices_to_remove = set()
                        mut_excl_items_to_check=[['in_front_of', 'behind'],
                                                 ['above', 'beneath'],
                                                ]
                        num_orig_spatial_rels+= len(obj_struct['spatial_relationship_scores'])
                        
                        for mut_exl_set in mut_excl_items_to_check:
                            if(self.check_all_in_items_exist(mut_exl_set, obj_struct['spatial_relationship'])):
                                max_rel_score_index = 0
                                max_rel_score=0
                                covered_indices = []
                                for rel_name in mut_exl_set:
                                    rel_index = obj_struct['spatial_relationship'].index(rel_name)
                                    covered_indices.append(rel_index)
                                    if(obj_struct['spatial_relationship_scores'][rel_index] > max_rel_score):
                                        max_rel_score_index = rel_index
                                        max_rel_score = obj_struct['spatial_relationship_scores'][rel_index]
                                less_score_indices = set(covered_indices) - set([max_rel_score_index])
                                rel_indices_to_remove = rel_indices_to_remove.union(less_score_indices)
                                #for rel_n in mut_exl_set:
                        obj_struct['spatial_relationship_scores'] = self.get_list_with_removed_elements(obj_struct['spatial_relationship_scores'], list(rel_indices_to_remove))
                        obj_struct['spatial_relationship'] = self.get_list_with_removed_elements(obj_struct['spatial_relationship'], list(rel_indices_to_remove))
                        num_spatial_rels_removed_total+=len(rel_indices_to_remove)

                    if('contacting_relationship_scores' in obj_struct):
                        mut_excl_items_to_check=[['holding', 'not_contacting'],
                                                 ['not_contacting', 'touching'],
                                                 ['sitting_on', 'standing_on'],
                                                 ['sitting_on', 'carrying'],
                                                 ['standing_on', 'carrying']
                                                ]
                        
                        rel_indices_to_remove = set()
                        num_orig_contact_rels+= len(obj_struct['contacting_relationship_scores'])
                        
                        
                        for mut_exl_set in mut_excl_items_to_check:
                            if(self.check_all_in_items_exist(mut_exl_set, obj_struct['contacting_relationship'])):
                                max_rel_score_index = 0
                                max_rel_score=0
                                covered_indices = []
                                for rel_name in mut_exl_set:
                                    rel_index = obj_struct['contacting_relationship'].index(rel_name)
                                    covered_indices.append(rel_index)
                                    if(obj_struct['contacting_relationship_scores'][rel_index] > max_rel_score):
                                        max_rel_score_index = rel_index
                                        max_rel_score = obj_struct['contacting_relationship_scores'][rel_index]
                                less_score_indices = set(covered_indices) - set([max_rel_score_index])
                                rel_indices_to_remove = rel_indices_to_remove.union(less_score_indices)
                                #for rel_n in mut_exl_set:
                        obj_struct['contacting_relationship_scores'] = self.get_list_with_removed_elements(obj_struct['contacting_relationship_scores'], list(rel_indices_to_remove))
                        obj_struct['contacting_relationship'] = self.get_list_with_removed_elements(obj_struct['contacting_relationship'], list(rel_indices_to_remove))
                        num_contact_rels_removed_total+=len(rel_indices_to_remove)

                    new_obj_struct_list.append(obj_struct)
                    
                or_obj_proposals_struct[video_id]['frames'][frame_id] = new_obj_struct_list
                
        print(f"Filtered {num_att_rels_removed_total} number of att relations from original {num_orig_att_rels} ({100*num_att_rels_removed_total/num_orig_att_rels:.2f}%)")
        print(f"Filtered {num_spatial_rels_removed_total} number of spatial relations from original {num_orig_spatial_rels} ({100*num_spatial_rels_removed_total/num_orig_spatial_rels:.2f}%)")
        print(f"Filtered {num_contact_rels_removed_total} number of contact relations from original {num_orig_contact_rels} ({100*num_contact_rels_removed_total/num_orig_contact_rels:.2f}%)")
        return or_obj_proposals_struct
    
    def filter_data_and_recompute_max_objs(self, 
                                           or_obj_proposals_struct,
                                           or_person_proposals_struct, 
                                           min_rel_score_thresh, 
                                           min_obj_score_thresh):
        num_att_rels_removed_total = 0
        num_spatial_rels_removed_total = 0
        num_contact_rels_removed_total = 0
        num_objs_removed_total = 0

        num_orig_total_objs = 1e-10 
        num_orig_att_rels = 1e-10
        num_orig_spatial_rels = 1e-10
        num_orig_contact_rels = 1e-10
        max_num_objs_for_frame = 0
        #or_obj_proposals_struct = deepcopy(or_obj_proposals_struct)
        #or_person_proposals_struct = deepcopy(or_person_proposals_struct)
        for video_id, video_struct in or_obj_proposals_struct.items():
            #print('hello')
            for frame_id, obj_struct_list in video_struct['frames'].items():
                obj_indices_to_remove = []
                new_obj_struct_list = []
                #print(len(obj_struct_list))
                #input()
                for obj_i, obj_struct in enumerate(obj_struct_list):
                    num_orig_total_objs+=1
                    if('obj_score' in obj_struct and obj_struct['obj_score'][0]<min_obj_score_thresh):
                        obj_indices_to_remove.append(obj_i)
                        num_objs_removed_total+=1

                    if('attention_relationship_scores' in obj_struct):
                        rel_indices_to_remove = []
                        for rel_i, rel_score in enumerate(obj_struct['attention_relationship_scores']):
                            num_orig_att_rels+=1
                            if(rel_score<min_rel_score_thresh):
                                rel_indices_to_remove.append(rel_i)
                                num_att_rels_removed_total+=1
                        obj_struct['attention_relationship_scores'] = self.get_list_with_removed_elements(obj_struct['attention_relationship_scores'], rel_indices_to_remove)
                        obj_struct['attention_relationship'] = self.get_list_with_removed_elements(obj_struct['attention_relationship'], rel_indices_to_remove)


                    if('spatial_relationship_scores' in obj_struct):
                        rel_indices_to_remove = []
                        for rel_i, rel_score in enumerate(obj_struct['spatial_relationship_scores']):
                            num_orig_spatial_rels+=1
                            if(rel_score<min_rel_score_thresh):
                                rel_indices_to_remove.append(rel_i)
                                num_spatial_rels_removed_total+=1
                        obj_struct['spatial_relationship_scores'] = self.get_list_with_removed_elements(obj_struct['spatial_relationship_scores'], rel_indices_to_remove)
                        obj_struct['spatial_relationship'] = self.get_list_with_removed_elements(obj_struct['spatial_relationship'], rel_indices_to_remove)

                    if('contacting_relationship_scores' in obj_struct):
                        rel_indices_to_remove = []
                        for rel_i, rel_score in enumerate(obj_struct['contacting_relationship_scores']):
                            num_orig_contact_rels+=1
                            if(rel_score<min_rel_score_thresh):
                                rel_indices_to_remove.append(rel_i)
                                num_contact_rels_removed_total+=1
                        #for rel_i_r in rel_indices_to_remove:
                        obj_struct['contacting_relationship_scores'] = self.get_list_with_removed_elements(obj_struct['contacting_relationship_scores'], rel_indices_to_remove)
                        obj_struct['contacting_relationship'] = self.get_list_with_removed_elements(obj_struct['contacting_relationship'], rel_indices_to_remove)
                    new_obj_struct_list.append(obj_struct)
                    
                #print(obj_indices_to_remove)
                #print(len(obj_struct_list))
                filt_obj_struct_list = self.get_list_with_removed_elements(new_obj_struct_list, obj_indices_to_remove)
                or_obj_proposals_struct[video_id]['frames'][frame_id] = filt_obj_struct_list
                max_num_objs_for_frame = max(len(filt_obj_struct_list), max_num_objs_for_frame)
                
        print(f"Filtered {num_att_rels_removed_total} number of att relations from original {num_orig_att_rels} ({100*num_att_rels_removed_total/num_orig_att_rels:.2f}%)")
        print(f"Filtered {num_spatial_rels_removed_total} number of spatial relations from original {num_orig_spatial_rels} ({100*num_spatial_rels_removed_total/num_orig_spatial_rels:.2f}%)")
        print(f"Filtered {num_contact_rels_removed_total} number of contact relations from original {num_orig_contact_rels} ({100*num_contact_rels_removed_total/num_orig_contact_rels:.2f}%)")
        print(f"Filtered {num_objs_removed_total} number of objects from original {num_orig_total_objs} ({100*num_objs_removed_total/num_orig_total_objs:.2f}%)")
        return or_obj_proposals_struct, or_person_proposals_struct, max_num_objs_for_frame

    def pad_obj_labels(self, obj_labels, max_num_objs, pad_token='<pad>'):
        return obj_labels + [self.obj_label2id[pad_token] for i in range(max_num_objs-len(obj_labels))]
    
    def pad_obj_rels(self, obj_rels, max_num_objs, rel_label_len):
        return obj_rels + [torch.zeros(rel_label_len).to(torch.float32) for i in range(max_num_objs-len(obj_rels))]
    
    
    def pad_obj_bboxes(self, bboxes, max_num_objs):
        return bboxes + [torch.zeros(4).to(torch.float32) for i in range(max_num_objs-len(bboxes))]
    
    def pad_obj_resolutions(self, resolutions, max_num_objs):
        return resolutions + [resolutions[0] for i in range(max_num_objs-len(resolutions))]
    
    def pad_sampled_frames(self, sampled_frames):
        return sampled_frames + [None for i in range(self.num_expected_sampled_frames - len(sampled_frames))]
    
    
    
    def _create_obj_and_rel_label_vocab(self, obj_label_vocab_path, relation_vocab_path_root):
        objects_vocab = {'<unk>':0, '<pad>':1}
        rels_vocab = {'<unk>':0, '<pad>':1}
        att_rels = {'<unk>':0, '<pad>':1}
        spatial_rels = {'<unk>':0, '<pad>':1}
        contacting_rels = {'<unk>':0, '<pad>':1}
        
        max_num_objs = 0
        
        for video_id in self.obj_proposals_struct:
            frame_structs = self.obj_proposals_struct[video_id]['frames']
            for frame_id in frame_structs:
                visible_objs=0
                for object_struct in frame_structs[frame_id]:
                    obj_label = object_struct['class']
                    if(object_struct['visible']):
                        visible_objs +=1
                    if(obj_label not in objects_vocab and obj_label is not None):
                        objects_vocab[obj_label] = len(objects_vocab)
                    if(object_struct['attention_relationship'] is not None):
                        for rel in object_struct['attention_relationship']:
                            if(rel not in rels_vocab):
                                rels_vocab[rel] = len(rels_vocab)
                            if(rel not in att_rels):
                                att_rels[rel] = len(att_rels)
                    if(object_struct['spatial_relationship'] is not None):
                        for rel in object_struct['spatial_relationship']:
                            if(rel not in rels_vocab):
                                rels_vocab[rel] = len(rels_vocab)
                            if(rel not in spatial_rels):
                                spatial_rels[rel] = len(spatial_rels)
                    if(object_struct['contacting_relationship'] is not None):
                        for rel in object_struct['contacting_relationship']:
                            if(rel not in rels_vocab):
                                rels_vocab[rel] = len(rels_vocab)
                            if(rel not in contacting_rels):
                                contacting_rels[rel] = len(contacting_rels)
                max_num_objs = max(max_num_objs, visible_objs)
                
        id2objlabel = {v:k for k,v in objects_vocab.items()}
        id2rellabel = {v:k for k,v in rels_vocab.items()}
        id2att_rel = {v:k for k,v in att_rels.items()}
        id2spatial_rel = {v:k for k,v in spatial_rels.items()}
        id2contacing_rel = {v:k for k,v in contacting_rels.items()}
        with open(obj_label_vocab_path,'wb') as f:
            pickle.dump((objects_vocab, id2objlabel, len(objects_vocab), max_num_objs),f)
        with open(os.path.join(relation_vocab_path_root, "star_combined_rel_labels.pkl"),'wb') as f:
            pickle.dump((rels_vocab, id2rellabel, len(rels_vocab)),f)
        with open(os.path.join(relation_vocab_path_root, "star_att_rel_labels.pkl"),'wb') as f:
            pickle.dump((att_rels, id2att_rel, len(att_rels)),f)
        with open(os.path.join(relation_vocab_path_root, "star_spatial_rel_labels.pkl"),'wb') as f:
            pickle.dump((spatial_rels, id2spatial_rel, len(spatial_rels)),f)
        with open(os.path.join(relation_vocab_path_root, "star_contact_rel_labels.pkl"),'wb') as f:
            pickle.dump((contacting_rels, id2contacing_rel, len(contacting_rels)),f)
            
    def get_dummy_obj_struct_for_padding(self):
        return [{'class': '<pad>',
              'bbox': (0.0, 0.0, 0.0, 0.0),
              'attention_relationship': [],
              'spatial_relationship': [],
              'contacting_relationship': [],
              #'metadata': {'tag': '0GH5O.mp4/doorway/000016', 'set': 'train'},
              'visible': True}]
    
    def get_dummy_padded_person_struct(self):
        return {'bbox': np.array([[ 0.0 ,  0.0, 0.0 , 0.0 ]],dtype=np.float32),
 'bbox_score': np.array([1.0],dtype=np.float32),
 'keypoints': np.array([[[ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.],
         [ 0.0 ,  0.0, 1.]]], dtype=np.float32),
}
    
    def __len__(self):
        return int(len(self.data) * self.train_percent)
    
    def __getitem__(self, index):
        item = self.data[int(index/self.train_percent)]
        video_id = item['video_id']
        og_q_id = item['question_id']
        question = item['question']
        if(self.without_labels):
            item['answer'] = '-1'
            item['question_program'] = ['-1']
        program = item['question_program']
        question_type = item['question_id'].split('_')[0]
        
        answer_text = item['answer'].strip().lower()
        choices_struct = item['choices']
        assert(len(choices_struct)==4)
        correct_choice_id = -1
        choice_texts = []
        choice_programs = []
        for ch_i, choice_struct in enumerate(choices_struct):
            choice_text = choice_struct['choice'].strip().lower()
            choice_texts.append(choice_text)
            if(choice_text == answer_text and not self.without_labels):
                correct_choice_id = ch_i
            
            if(not self.without_labels):
                choice_programs.append(choice_struct['choice_program'])
            else:
                choice_programs.append(['None'])
                
        #if(correct_choice_id not in range(0,4)):
         #   print(f"couldn't find choice {choice_texts} and answer [{answer_text}] with choice id {correct_choice_id}")
        if(not self.without_labels):
            assert(correct_choice_id in range(0,4))
        #answer_id = self.label2id[answer_text]
            target = torch.zeros(4)
            target[correct_choice_id] = 1
        else:
            target = torch.zeros(4)
        data_id = int(index/self.train_percent)
        frames_objs_dict = self.obj_proposals_struct[f'{video_id}.mp4']
        frames_person_dict = self.person_proposals_struct[f'{video_id}.mp4']
        assert(frames_objs_dict['vid_info']['resolution'] == frames_person_dict['vid_info']['resolution'])
        assert(frames_objs_dict['frames'].keys() == frames_person_dict['frames'].keys())
        
        vid_h, vid_w = frames_objs_dict['vid_info']['resolution'] 
        #num_frames = len(frames_objs_dict['frames'])
        
        frames_objs_info = frames_objs_dict['frames'] 
        frames_person_info = frames_person_dict['frames'] 
        applicable_frames = self.q_to_vidid_and_frames_dict[og_q_id][video_id]
        frames_objs_keys = sorted([frame+'.png' for frame in applicable_frames])
        final_keys = []
        #print(frames_objs_info.keys(), f
        for frame_key in frames_objs_keys:
            if(frame_key not in frames_objs_info or frame_key not in frames_person_info):
                print(f"warning: frame key {frame_key} not found in list of keys for video id {video_id}")
                #print(frame_key, list(frames_objs_info.keys())[:150])
                #input()
                
            else:
                final_keys.append(frame_key)
        frames_objs_keys = final_keys
        #frames_objs_keys = sorted(frames_objs_info.keys())
        num_frames = len(frames_objs_keys)
        
        
       # get_iteratively_sampled_indices(num_frames_to_sample, num_given_frames, verbose=False, return_sorted=True):
        sampled_frame_indices, _ = get_iteratively_sampled_indices(self.num_sample_frames, num_frames)#, self.#get_iteratively_sampled_frames(num_frames, ) proposal_struct['frames'][::self.frame_sample_every_n]
        num_initially_sampled_frames = len(sampled_frame_indices)
        num_nonpadded_frames = num_initially_sampled_frames
        if(len(sampled_frame_indices)<self.num_sample_frames):
            if(self.print_warnings):
                print(f"Warning: num sampled frames {num_initially_sampled_frames} < preset sample frame num {self.num_sample_frames}")
            sampled_frame_indices = sampled_frame_indices + [None for i_ in range(self.num_sample_frames-len(sampled_frame_indices))]
            #print(0, num_nonpadded_frames)
            #num_nonpadded_frames = 
        elif(len(sampled_frame_indices)>self.num_sample_frames):
            if(self.print_warnings):
                print(f"Warning: num sampled frames {num_initially_sampled_frames} > preset sample frame num {self.num_sample_frames}")
            sampled_frame_indices = sampled_frame_indices[:self.num_sample_frames]
            num_nonpadded_frames = self.num_sample_frames
            #print(1, num_nonpadded_frames)
        item_num_max_objs = 0
        
        frames_objs_bboxes = []
        frames_person_bboxes = []
        frames_person_pose_coords =[]
        frames_objs_labels = []
        frames_objs_rels = []
        frames_objs_att_rels = []
        frames_objs_spatial_rels = []
        frames_objs_contacting_rels = []
        
        frames_num_objs = []
        frames_resolutions = []
        sampled_frame_og_ids = []
        
        for sampled_frame_id in sampled_frame_indices:
            if(sampled_frame_id is None):
                frame_key = -1
                obj_structs = self.get_dummy_obj_struct_for_padding()
                person_struct = self.get_dummy_padded_person_struct()
                sampled_frame_og_ids.append(None)
            
            else:
                frame_key = frames_objs_keys[sampled_frame_id]
                #print(1, frame_key, og_q_id)
                #input()
                #print(frame_key)
                #print(frames_objs_info.keys())
                obj_structs = frames_objs_info[frame_key] # = frames_objs_dict['frames'] 
                person_struct = frames_person_info[frame_key]
                sampled_frame_og_ids.append(frame_key)
            
                
            objs_bboxes = []
            person_bboxes = []
            person_pose_coords = []
            objs_labels = []
            objs_rels = []
            objs_att_rels = []
            objs_spatial_rels = []
            objs_contacting_rels = []
            #objs_resolutions = []
            #try:
             #   x1,y1,x2,y2 = person_struct['bbox'][0]
                #print(person_struct['bbox'])
                #input()
            #except:
             #   print('ERROR person bbox', person_struct['bbox'])
              #  x1,y1,x2,y2 = (0,0,0,0)
            if(person_struct['bbox'].shape[0]==0):
                #print(f"Warning: no person bbox found for frame id {sampled_frame_id} for video id {video_id}")
                x1,y1,x2,y2 = (0,0,0,0)
            else:
                x1,y1,x2,y2 = person_struct['bbox'][0]
            person_bboxes.append(torch.tensor([x1,y1,x2,y2]).to(torch.float32))
            if(person_struct['keypoints'].shape[0]==0):
                person_pose_coords.append(torch.tensor(torch.zeros(person_struct['keypoints'].shape[1:])).to(torch.float32))
            else:
                person_pose_coords.append(torch.tensor(person_struct['keypoints'][0]))
                
                
            
            obj_i=0
            for obj_struct in obj_structs:
                if(not obj_struct['visible']):
                    continue
                objs_labels.append(self.obj_label2id[obj_struct['class']])
                x,y,w,h = obj_struct['bbox']
                objs_bboxes.append(torch.tensor([x, y, x+w, y+h]).to(torch.float32))
                
                
                objs_rels.append(torch.zeros(self.combined_rel_label_len))
                objs_att_rels.append(torch.zeros(self.att_rel_label_len))
                objs_spatial_rels.append(torch.zeros(self.spatial_rel_label_len))
                objs_contacting_rels.append(torch.zeros(self.contact_rel_label_len))
                
                 
                for rel in obj_struct['attention_relationship']:
                    rel_id = self.rel_label2id[rel]
                    #print('A', rel_id, objs_rels[obj_i])
                    objs_rels[obj_i][rel_id] = 1.0
                    objs_att_rels[obj_i][self.att_rel_label2id[rel]] = 1.0
                #except:
                 #   print(len(objs_rels), obj_i, 
                for rel in obj_struct['spatial_relationship']:
                    rel_id = self.rel_label2id[rel]
                    objs_rels[obj_i][rel_id] = 1.0
                    objs_spatial_rels[obj_i][self.spatial_rel_label2id[rel]] = 1.0
                    
                for rel in obj_struct['contacting_relationship']:
                    rel_id = self.rel_label2id[rel]
                    objs_rels[obj_i][rel_id] = 1.0
                    objs_contacting_rels[obj_i][self.contact_rel_label2id[rel]] = 1.0
                    
                obj_i+=1
                
                
            num_objs = len(objs_labels)
            item_num_max_objs = max(item_num_max_objs, num_objs)
            objs_bboxes = self.pad_obj_bboxes(objs_bboxes, self.max_num_objs)
            
            objs_rels = self.pad_obj_rels(objs_rels, self.max_num_objs, rel_label_len= self.combined_rel_label_len)
            objs_att_rels = self.pad_obj_rels(objs_att_rels, self.max_num_objs, rel_label_len= self.att_rel_label_len)
            objs_spatial_rels = self.pad_obj_rels(objs_spatial_rels, self.max_num_objs, rel_label_len= self.spatial_rel_label_len)
            objs_contacting_rels = self.pad_obj_rels(objs_contacting_rels, self.max_num_objs, rel_label_len=self.contact_rel_label_len )
            
            objs_labels = self.pad_obj_labels(objs_labels, self.max_num_objs)
            
            frames_person_bboxes.append(torch.stack(person_bboxes))
            frames_person_pose_coords.append(torch.stack(person_pose_coords))
            
            frames_objs_bboxes.append(torch.stack(objs_bboxes))
            frames_objs_rels.append(torch.stack(objs_rels))
            frames_objs_att_rels.append(torch.stack(objs_att_rels))
            frames_objs_spatial_rels.append(torch.stack(objs_spatial_rels))
            frames_objs_contacting_rels.append(torch.stack(objs_contacting_rels))
            
            frames_objs_labels.append(torch.tensor(objs_labels))
            
            frames_num_objs.append(num_objs)
            frames_resolutions.append(torch.tensor([vid_h, vid_w]))
            
            
            
        frames_person_bboxes = torch.stack(frames_person_bboxes)
        frames_person_pose_coords = torch.stack(frames_person_pose_coords)
        frames_objs_bboxes = torch.stack(frames_objs_bboxes)
        frames_objs_rels = torch.stack(frames_objs_rels)
        frames_objs_att_rels = torch.stack(frames_objs_att_rels)
        frames_objs_spatial_rels = torch.stack(frames_objs_spatial_rels)
        frames_objs_contacting_rels = torch.stack(frames_objs_contacting_rels)
        
        frames_objs_labels = torch.stack(frames_objs_labels)
        frames_resolutions = torch.stack(frames_resolutions)
        
        return question, answer_text, choice_texts, target, frames_person_bboxes, frames_person_pose_coords, frames_objs_bboxes, frames_objs_rels, frames_objs_att_rels, frames_objs_spatial_rels, frames_objs_contacting_rels, frames_objs_labels, frames_num_objs, frames_resolutions, program, choice_programs, question_type, og_q_id, int(index/self.train_percent), item_num_max_objs, num_nonpadded_frames, video_id, sampled_frame_og_ids
    
    
        
        
def collate_fn_star(batch):
    """
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    """
    sorted_by_len = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    batch = {'questions':[],
             'answer_texts':[],
             'choice0_texts':[],
             'choice1_texts':[],
             'choice2_texts':[],
             'choice3_texts':[],
             'answer_ids':[],
             'person_bbox':[],
             'person_pose_coords':[],
             'obj_bboxes':[],
             'obj_rels':[],
             'obj_att_rels':[],
             'obj_spatial_rels':[],
             'obj_contacting_rels':[],
             'obj_labels':[],
             'num_objs':[],
             'frame_resolutions':[],
             'programs':[],
             'choice0_programs':[],
             'choice1_programs':[],
             'choice2_programs':[],
             'choice3_programs':[],
             'q_data_index':[],
             'og_q_ids':[],
             'q_types':[],
             'num_nonpadded_frames':[],
             'video_ids':[],
             'sampled_frame_og_ids':[],
            }
    
    for i, b in enumerate(sorted_by_len):
        question, answer_text, choice_texts, target, frames_person_bboxes, frames_person_pose_coords, frames_objs_bboxes, frames_objs_rels, frames_objs_att_rels, frames_objs_spatial_rels, frames_objs_contacting_rels, frames_objs_labels, frames_num_objs, frames_resolutions, program, choice_programs, question_type, og_q_id, q_data_index, item_num_max_objs, num_nonpadded_frames, video_id, sampled_frame_og_ids = b
        
        batch['questions'].append(question)
        batch['answer_texts'].append(answer_text)
        batch['choice0_texts'].append(choice_texts[0])
        batch['choice1_texts'].append(choice_texts[1])
        batch['choice2_texts'].append(choice_texts[2])
        batch['choice3_texts'].append(choice_texts[3])
        
        batch['answer_ids'].append(target)
        batch['person_bbox'].append(frames_person_bboxes)
        batch['person_pose_coords'].append(frames_person_pose_coords)
        batch['obj_bboxes'].append(frames_objs_bboxes)
        
        batch['obj_rels'].append(frames_objs_rels)
        batch['obj_att_rels'].append(frames_objs_att_rels)
        batch['obj_spatial_rels'].append(frames_objs_spatial_rels)
        batch['obj_contacting_rels'].append(frames_objs_contacting_rels)
        
        batch['obj_labels'].append(frames_objs_labels)
        batch['num_objs'].append(frames_num_objs)
        batch['frame_resolutions'].append(frames_resolutions)
        batch['programs'].append(program)
        batch['choice0_programs'].append(choice_programs[0])
        batch['choice1_programs'].append(choice_programs[1])
        batch['choice2_programs'].append(choice_programs[2])
        batch['choice3_programs'].append(choice_programs[3])
        batch['q_types'].append(question_type)
        batch['q_data_index'].append(q_data_index)
        batch['og_q_ids'].append(og_q_id)
        batch['num_nonpadded_frames'].append(num_nonpadded_frames)
        batch['video_ids'].append(video_id)
        batch['sampled_frame_og_ids'].append(sampled_frame_og_ids)
        
    batch['answer_ids'] = torch.stack(batch['answer_ids'])
    batch['person_bbox'] = torch.stack(batch['person_bbox'])
    batch['person_pose_coords'] = torch.stack(batch['person_pose_coords'])
    batch['obj_bboxes'] = torch.stack(batch['obj_bboxes'])
    batch['obj_rels'] = torch.stack(batch['obj_rels'])
    batch['obj_att_rels'] = torch.stack(batch['obj_att_rels'])
    batch['obj_spatial_rels'] = torch.stack(batch['obj_spatial_rels'])
    batch['obj_contacting_rels'] = torch.stack(batch['obj_contacting_rels'])
    
    batch['obj_labels'] = torch.stack(batch['obj_labels'])
    batch['frame_resolutions'] = torch.stack(batch['frame_resolutions'])
    
    return batch  

def collate_fn_batch_wise_max_objs(batch):
    """
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    """
    sorted_by_len = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    bwise_max_num_objs = 0
    for b in batch:
        bwise_max_num_objs = max(bwise_max_num_objs, b[-1])
        
    sorted_by_len = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    batch = {'questions':[],
             'answer_texts':[],
             'choice0_texts':[],
             'choice1_texts':[],
             'choice2_texts':[],
             'choice3_texts':[],
             'answer_ids':[],
             'person_bbox':[],
             'person_pose_coords':[],
             'obj_bboxes':[],
             'obj_rels':[],
             'obj_att_rels':[],
             'obj_spatial_rels':[],
             'obj_contacting_rels':[],
             'obj_labels':[],
             'num_objs':[],
             'frame_resolutions':[],
             'programs':[],
             'choice0_programs':[],
             'choice1_programs':[],
             'choice2_programs':[],
             'choice3_programs':[],
             'q_data_index':[],
             'og_q_ids':[],
             'q_types':[],
             'num_nonpadded_frames':[],
             'sampled_frame_og_ids':[],
            }
    
    for i, b in enumerate(sorted_by_len):
        question, answer_text, choice_texts, target, frames_person_bboxes, frames_person_pose_coords, frames_objs_bboxes, frames_objs_rels, frames_objs_att_rels, frames_objs_spatial_rels, frames_objs_contacting_rels, frames_objs_labels, frames_num_objs, frames_resolutions, program, choice_programs, question_type, og_q_id, q_data_index, item_num_max_objs, num_nonpadded_frames, video_id, sampled_frame_og_ids = b
        
        frames_objs_bboxes = frames_objs_bboxes[:,:bwise_max_num_objs,:]
        frames_objs_rels = frames_objs_rels[:,:bwise_max_num_objs,:,:]
        frames_objs_att_rels = frames_objs_att_rels[:,:bwise_max_num_objs,:,:]
        frames_objs_spatial_rels = frames_objs_spatial_rels[:,:bwise_max_num_objs,:,:]
        frames_objs_contacting_rels = frames_objs_contacting_rels[:,:bwise_max_num_objs,:,:]
        
        frames_objs_labels = frames_objs_labels[:,:bwise_max_num_objs]
        
        
        batch['questions'].append(question)
        batch['answer_texts'].append(answer_text)
        batch['choice0_texts'].append(choice_texts[0])
        batch['choice1_texts'].append(choice_texts[1])
        batch['choice2_texts'].append(choice_texts[2])
        batch['choice3_texts'].append(choice_texts[3])
        
        batch['answer_ids'].append(target)
        batch['person_bbox'].append(frames_person_bboxes)
        batch['person_pose_coords'].append(frames_person_pose_coords)
        batch['obj_bboxes'].append(frames_objs_bboxes)
        
        batch['obj_rels'].append(frames_objs_rels)
        batch['obj_att_rels'].append(frames_objs_att_rels)
        batch['obj_spatial_rels'].append(frames_objs_spatial_rels)
        batch['obj_contacting_rels'].append(frames_objs_contacting_rels)
        
        batch['obj_labels'].append(frames_objs_labels)
        batch['num_objs'].append(frames_num_objs)
        batch['frame_resolutions'].append(frames_resolutions)
        batch['programs'].append(program)
        batch['choice0_programs'].append(choice_programs[0])
        batch['choice1_programs'].append(choice_programs[1])
        batch['choice2_programs'].append(choice_programs[2])
        batch['choice3_programs'].append(choice_programs[3])
        batch['q_types'].append(question_type)
        batch['q_data_index'].append(q_data_index)
        batch['og_q_ids'].append(og_q_id)
        batch['num_nonpadded_frames'].append(num_nonpadded_frames)
        batch['video_ids'].append(video_id)
        batch['sampled_frame_og_ids'].append(sampled_frame_og_ids)
        
        
    batch['answer_ids'] = torch.stack(batch['answer_ids'])
    batch['person_bbox'] = torch.stack(batch['person_bbox'])
    batch['person_pose_coords'] = torch.stack(batch['person_pose_coords'])
    batch['obj_bboxes'] = torch.stack(batch['obj_bboxes'])
    batch['obj_rels'] = torch.stack(batch['obj_rels'])
    batch['obj_rels'] = torch.stack(batch['obj_rels'])
    batch['obj_att_rels'] = torch.stack(batch['obj_att_rels'])
    batch['obj_spatial_rels'] = torch.stack(batch['obj_spatial_rels'])
    batch['obj_contacting_rels'] = torch.stack(batch['obj_contacting_rels'])
    
    batch['obj_labels'] = torch.stack(batch['obj_labels'])
    batch['frame_resolutions'] = torch.stack(batch['frame_resolutions'])
    
    return batch  

    
        
        
        
        