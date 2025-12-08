import wandb
import sys
import os
import pickle
from collections import Counter
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
import json
import random
from dataset import get_dataloader
from config import get_default_cfg, config_to_wandb
import torch
from lavis.models import load_model_and_preprocess
from vlm_model import VLM_for_STAR
from eval_utils import update_program_type_and_question_wise_acc


seed_val = 200 #91  # 6#192 #6#191#6 (best) #93 #42
# Set a random seed for PyTorch
torch.manual_seed(seed_val)

# Set a random seed for the random module (used in data loading, etc.)
random.seed(seed_val)

# Set a random seed for NumPy (used in data preprocessing, etc.)
np.random.seed(seed_val)


device = None

def convert_set_to_torch_binary_feat(set_value_indices, max_value_len):
    #assume set_value_indices is zero indexed
    array = torch.zeros(max_value_len, dtype=torch.float32)
    array[list(set_value_indices)] = 1
    return array

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, train_dataloader, ds_type='og', txt_processors=None, do_answer_argmax=True):
    dataloader = iter(train_dataloader)
    pbar = tqdm(dataloader)
    moving_accuracy = None
    moving_steps = None
    train_accuracy_dict = None
    net.train(True)
    for batch_n, item in enumerate(pbar):
        person_bbox_x, person_keypoints_x = item['person_bbox'], item['person_pose_coords']
        obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x = item['obj_bboxes'], item['obj_labels'], item['obj_rels'], item['obj_att_rels'], item['obj_spatial_rels'], item['obj_contacting_rels']
        
        lang_question_x = [txt_processors['eval'](qs) for qs in item['questions']]
        lang_choice0_x = [txt_processors['eval'](qs) for qs in item['choice0_texts']]
        lang_choice1_x = [txt_processors['eval'](qs) for qs in item['choice1_texts']]
        lang_choice2_x = [txt_processors['eval'](qs) for qs in item['choice2_texts']]
        lang_choice3_x = [txt_processors['eval'](qs) for qs in item['choice3_texts']]
                
        
        
        frame_resolutions = item['frame_resolutions']
        num_objs = item['num_objs']
        num_unpadded_frames = item['num_nonpadded_frames'] 
        answer = item['answer_ids']
        #answer = answer.argmax(1)

        person_bbox_x, person_keypoints_x, obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x, frame_resolutions, answer = (
            person_bbox_x.to(device),
            person_keypoints_x.to(device),
            obj_bbox_x.to(device),
            obj_label_x.to(device),
            combined_rel_x.to(device),
            att_rel_x.to(device),
            spatial_rel_x.to(device),
            contact_rel_x.to(device),
            frame_resolutions.to(device),
            answer.to(device)
        )
        
        
        net.zero_grad()
        
        output = net((person_bbox_x, person_keypoints_x),
                     (obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x),
                     (lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x),
                     frame_resolutions,
                     num_objs, 
                     num_unpadded_frames,
                     save_atts=False)
    
        if(do_answer_argmax):
            answer_t = answer.argmax(1)
        else:
            answer_t = answer
        loss = criterion(output, answer_t)

        loss.backward()

        if cfg.SOLVER.GRAD_CLIP:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.GRAD_CLIP)
        optimizer.step()

        # train metrics
        train_accuracy_dict = update_program_type_and_question_wise_acc(item,
                                             answer.detach().argmax(1),
                                             output.detach().argmax(1),
                                             accuracy_dict= train_accuracy_dict,
                                             batch_number=batch_n)
        #print(train_accuracy_dict)
        #input()
        #if(batch_n>5):
         #   break
        correct = output.detach().argmax(1) == answer.detach().argmax(1)
        accuracy = correct.float().mean().item()
        if moving_accuracy is None:
            moving_accuracy = accuracy
            moving_loss = loss.item()
        else:
            moving_accuracy = moving_accuracy * 0.99 + accuracy * 0.01
            moving_loss = moving_loss * 0.99 + loss.item() * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch, loss.item(), moving_accuracy
            )
        )

        accumulate(net_running, net)
        #if(batch_n>5):
         #   break
        #break
    
    wandb.log({
        f"train_{ds_type}_precision": moving_accuracy,
        f"train_{ds_type}_Overall_acc": train_accuracy_dict['overall']['avg_acc'],
        f"train_{ds_type}_Feasibility_acc": train_accuracy_dict['by_program_type']['feasibility']['indv']['avg_acc'] if 'feasibility' in train_accuracy_dict['by_program_type'] else -1, 
        f"train_{ds_type}_Interaction_acc": train_accuracy_dict['by_program_type']['interaction']['indv']['avg_acc'] if 'interaction' in train_accuracy_dict['by_program_type'] else -1, 
        f"train_{ds_type}_Prediction_acc": train_accuracy_dict['by_program_type']['prediction']['indv']['avg_acc'] if 'prediction' in train_accuracy_dict['by_program_type'] else -1, 
        f"train_{ds_type}_Sequence_acc": train_accuracy_dict['by_program_type']['sequence']['indv']['avg_acc'] if 'sequence' in train_accuracy_dict['by_program_type'] else -1,
        "epoch": epoch
        })
    
    return accuracy, train_accuracy_dict



def valid(epoch, val_dataloaders, test_only=False, ds_type='og', txt_processors=None, apply_scheduler_at_val_set_i=0, start_scheduler_epoch=0):
    results = []
    val_accuracy_dict = None
    val_dict_outs = []
    for val_i, val_dataloader in enumerate(val_dataloaders):
        dataloader = iter(val_dataloader)
        pbar = tqdm(dataloader)
        
        net_running.train(False)
        with torch.no_grad():
            all_corrects = 0
            for i, item in enumerate(pbar):
                person_bbox_x, person_keypoints_x = item['person_bbox'], item['person_pose_coords']
                obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x = item['obj_bboxes'], item['obj_labels'], item['obj_rels'], item['obj_att_rels'], item['obj_spatial_rels'], item['obj_contacting_rels']

                lang_question_x = [txt_processors['eval'](qs) for qs in item['questions']]
                lang_choice0_x = [txt_processors['eval'](qs) for qs in item['choice0_texts']]
                lang_choice1_x = [txt_processors['eval'](qs) for qs in item['choice1_texts']]
                lang_choice2_x = [txt_processors['eval'](qs) for qs in item['choice2_texts']]
                lang_choice3_x = [txt_processors['eval'](qs) for qs in item['choice3_texts']]


                frame_resolutions = item['frame_resolutions']
                num_objs = item['num_objs']
                num_unpadded_frames = item['num_nonpadded_frames'] 
                answer = item['answer_ids']
                
                person_bbox_x, person_keypoints_x, obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x, frame_resolutions, answer = (
                    person_bbox_x.to(device),
                    person_keypoints_x.to(device),
                    obj_bbox_x.to(device),
                    obj_label_x.to(device),
                    combined_rel_x.to(device),
                    att_rel_x.to(device),
                    spatial_rel_x.to(device),
                    contact_rel_x.to(device),
                    frame_resolutions.to(device),
                    answer.to(device)
                )

                output  = net_running((person_bbox_x, person_keypoints_x),
                                     (obj_bbox_x, obj_label_x, combined_rel_x, att_rel_x, spatial_rel_x, contact_rel_x),
                                     (lang_question_x, lang_choice0_x, lang_choice1_x, lang_choice2_x, lang_choice3_x),
                                     frame_resolutions,
                                     num_objs, 
                                     num_unpadded_frames,
                                     save_atts=False)
                answer = answer.argmax(1)
                correct = output.detach().argmax(1) == answer
                val_accuracy_dict = update_program_type_and_question_wise_acc(item,
                                             answer.detach(),
                                             output.detach().argmax(1),
                                             accuracy_dict= val_accuracy_dict,
                                             batch_number=i)
                #print(val_accuracy_dict)
                #input()
                #if(i>5):
                 #   break
                all_corrects += correct.float().mean().item()
                
                pbar.set_description(
                    'Val set {}; Epoch: {}; Acc: {:.5f}'.format(val_i, 
                        epoch, all_corrects / (i + 1)
                    )
                )
                    
            if(not test_only):   
                wandb.log({
                    f"valid_{ds_type}_{val_i}_precision": all_corrects / len(dataloader),
                    f"valid_{ds_type}_overall_acc": val_accuracy_dict['overall']['avg_acc'],
                    f"valid_{ds_type}_Feasibility_acc": val_accuracy_dict['by_program_type']['feasibility']['indv']['avg_acc'] if 'feasibility' in val_accuracy_dict['by_program_type'] else -1, 
                    f"valid_{ds_type}_Interaction_acc": val_accuracy_dict['by_program_type']['interaction']['indv']['avg_acc'] if 'interaction' in val_accuracy_dict['by_program_type'] else -1, 
                    f"valid_{ds_type}_Prediction_acc": val_accuracy_dict['by_program_type']['prediction']['indv']['avg_acc'] if 'prediction' in val_accuracy_dict['by_program_type'] else -1, 
                    f"valid_{ds_type}_Sequence_acc": val_accuracy_dict['by_program_type']['sequence']['indv']['avg_acc'] if 'sequence' in val_accuracy_dict['by_program_type'] else -1,
                    "epoch": epoch
                    })

            if scheduler and val_i == apply_scheduler_at_val_set_i and epoch>start_scheduler_epoch:
                scheduler.step(all_corrects / len(dataloader))

            results.append(all_corrects / len(dataloader))
            val_dict_outs.append(val_accuracy_dict)
    return results, val_dict_outs

def generate_cfg():
    parser = argparse.ArgumentParser(description="STAR-VideoQA training/eval script")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--test-only",
        type=bool,
        default=False
    )
    
    parser.add_argument(
        "--load-path",
        type=str,
        default=''
    )
    
    
    
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = get_default_cfg(mode='star')

    # update config with configuration file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if(cfg.LOAD_PATH in ['',' ',False,None] and not cfg.LOAD): #if load not specified in config but specified in args, then update in cfg
        cfg.LOAD_PATH = args.load_path
    if(args.load_path not in ['', False,' '] and not cfg.LOAD):
        cfg.LOAD=True
    cfg.freeze()
    return cfg, args.test_only

if __name__ == '__main__':
    cfg, test_only = generate_cfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print("using {}".format(device))
    
    #save_model_every_eps = []#20,25,30,40,50]

    
    if(not test_only):
        wandb.init(project=cfg.WANDB.PROJECT_NAME, config=config_to_wandb(cfg))
        if cfg.WANDB.EXPERIMENT_NAME:
            wandb.run.name = cfg.WANDB.EXPERIMENT_NAME

    train_dataloader, val_dataloaders = get_dataloader(cfg,
                                                         )
    
    net = VLM_for_STAR(cfg).to(device)
    net_running = VLM_for_STAR(cfg).to(device)
    if cfg.LOAD:
        print("Loading model from {}".format(cfg.LOAD_PATH))
        with open(cfg.LOAD_PATH, 'rb') as f:
            state = torch.load(f, map_location=device)
        net.load_state_dict(state, strict=False)
    accumulate(net_running, net, 0)

    print(net)
    #print(f'num time steps: {net.vlm_module.max_computation_steps}')
    do_answer_argmax_for_loss = True
    if(cfg.SOLVER.CRITERION in ['cross_entropy']):
        criterion = nn.CrossEntropyLoss()
        do_answer_argmax_for_loss = True
    elif(cfg.SOLVER.CRITERION in ['binary_cross_entropy']):
        criterion = nn.BCEWithLogitsLoss()
        do_answer_argmax_for_loss = False
    else:
        raise ValueError(f"SOLVER.CRITERION {cfg.SOLVER.CRITERION} not found")
    
    if cfg.SOLVER.LANG_ENC_LR:
        #filtered = ['actmac.gate.weight', 'actmac.gate.bias']
        print(f"Using 2 diff LRs: lang_enc_lr {cfg.SOLVER.LANG_ENC_LR} and others_lr {cfg.SOLVER.LR}")
        
        lang_enc_params = map(lambda kv: kv[1], (filter(lambda kv: 'lang_encoder' in kv[0],
                    net.named_parameters())))
        base_params = map(lambda kv: kv[1], (filter(lambda kv: 'lang_encoder' not in kv[0],
                    net.named_parameters())))
        optimizer = optim.Adam([{
                                    'params': base_params
                                }, {
                                    'params': lang_enc_params,
                                    'lr': cfg.SOLVER.LANG_ENC_LR
                                }],
                                lr=cfg.SOLVER.LR)
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)

    # LR scheduler
    scheduler = None
    if cfg.SOLVER.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.SOLVER.FACTOR, patience=cfg.SOLVER.PATIENCE, threshold=0.001, threshold_mode='rel')

    start_scheduler_epoch = cfg.SOLVER.START_SCHEDULER_EPOCH
    
    best_model_save_path = os.path.join(cfg.SAVE_DIRECTORY,'best_model.model')
    
    best_result = -1
    _, _, txt_processors = load_model_and_preprocess(name='blip_vqa', model_type='vqav2', is_eval=True, device='cpu') 
    del(_)
    
    if(not(os.path.exists(cfg.SAVE_DIRECTORY)) and not test_only):
        os.mkdir(cfg.SAVE_DIRECTORY)
    
    ds_type = 'star'
    
    for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
        if(epoch==1 and not test_only):
            with open(os.path.join(cfg.SAVE_DIRECTORY, 'model_config.json'), 'w') as f:
                json.dump(cfg, f, indent=4) 

        is_best=False
        if(not test_only):
            results_tr, train_acc_dict = train(epoch, train_dataloader, txt_processors = txt_processors, ds_type = ds_type, do_answer_argmax=do_answer_argmax_for_loss)

        results, val_acc_dicts = valid(epoch, val_dataloaders, test_only=test_only, txt_processors = txt_processors, apply_scheduler_at_val_set_i=0, ds_type = ds_type, start_scheduler_epoch=start_scheduler_epoch)
        
        if(val_acc_dicts[0]['overall']['avg_acc']>best_result):
            best_result = val_acc_dicts[0]['overall']['avg_acc'] #results[0]
            is_best=True
            print(f"best result is now: {best_result}")


        if(test_only):
            json.dump(val_acc_dicts[0], open(os.path.join(cfg.SAVE_DIRECTORY, 'testonly_val_acc_dict.json'),'w'), indent=4)
            break
        if(is_best):
            with open(best_model_save_path, 'wb') as f:
                torch.save(net_running.state_dict(), f)
            json.dump(train_acc_dict, open(os.path.join(cfg.SAVE_DIRECTORY, 'best_train_acc_dict.json'),'w'), indent=4)
            json.dump(val_acc_dicts[0], open(os.path.join(cfg.SAVE_DIRECTORY, 'best_val_acc_dict.json'),'w'), indent=4)
