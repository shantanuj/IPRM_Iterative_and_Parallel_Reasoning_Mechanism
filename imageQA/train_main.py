#Example usage: python3 train.py --mode clevr-cogen --config-file=configs/CLEVR/MAC/mac_daft_12_clevrcogen.yaml
#for test only: CUDA_VISIBLE_DEVICES=0 python3 train.py --mode clevr-cogen --config-file=configs/CLEVR/MAC/mac_daft_12_clevrcogen.yaml --test-only True --load-path="/data/usrdata/shan/MAC_CLEVR_models/mac_12_daft_no_prev_control_cogen/checkpoint_mac12_daft_no_prev_control_cogen_40.model"
#CUDA_VISIBLE_DEVICES=2 python3 train_with_blip_tokenizer_iprm_models.py --config-file="configs/CLEVR/MAC/000_multi-token-with-blip-tokenizer-op-and-res-init-vis-projs.yaml" --mode clevr --model-type mac
#CUDA_VISIBLE_DEVICES=0 python3 train_iprm_main.py --config-file="configs/CLEVR/MAC/iprm_config.yaml" --mode clevr --model-type mac
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
from vlm_model import VLM_for_CLEVR

seed_val = 93
torch.manual_seed(seed_val)

random.seed(seed_val)

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


def train(epoch, train_dataloader, ds_type=None, txt_processors=None, max_steps=None):
    dataloader = iter(train_dataloader)
    pbar = tqdm(dataloader)
    moving_accuracy = None
    moving_loss = None
    moving_steps = None

    net.train(True)
    count=0
    for item in pbar:
        
        image, question, answer = item['images'], [txt_processors['eval'](qs) for qs in item['questions']], item['answer_ids']
        q_len=None
        answer = answer.argmax(1)
        
        image, answer = (
            image.to(device),
            answer.to(device),
        )
        #if()
        
        net.zero_grad()
        
        output = net(image, question, save_atts=False)
        
        loss = criterion(output, answer)

        loss.backward()

        if cfg.SOLVER.GRAD_CLIP:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.GRAD_CLIP)
        optimizer.step()

        # train metrics
        correct = output.detach().argmax(1) == answer
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
        count+=1
        if(max_steps is not None and count>=max_steps):
            break
        
    if(ds_type is not None):
        wandb.log({
            "train_{}_precision".format(ds_type): moving_accuracy,
            })
    else:
        wandb.log({
            "train_og_precision": moving_accuracy,
            })


def valid(epoch, val_dataloaders, test_only=False, ds_type=None, txt_processors=None, apply_scheduler_at_val_set_i=0, start_scheduler_epoch=0, max_steps=None):
    results = []
    for val_i, val_dataloader in enumerate(val_dataloaders):
        dataloader = iter(val_dataloader)
        pbar = tqdm(dataloader)
        
        net_running.train(False)
        with torch.no_grad():
            all_corrects = 0
            count=0
            for i, item in enumerate(pbar):
                if(max_steps is not None and count>=max_steps):
                    break
                count+=1
                image, question, answer = item['images'], [txt_processors['eval'](qs) for qs in item['questions']], item['answer_ids']
                q_len=None
                answer = answer.argmax(1)

                image, answer = (
                    image.to(device),
                    answer.to(device),
                )

                output  = net_running(image, question, save_atts=False)
                correct = output.detach().argmax(1) == answer.to(device)

                all_corrects += correct.float().mean().item()
                
                pbar.set_description(
                    'Val set {}; Epoch: {}; Acc: {:.5f}'.format(val_i, 
                        epoch, all_corrects / (i + 1)
                    )
                )
                    

            if(not test_only):   
                if(ds_type is not None):
                    wandb.log({
                        "valid_{}_{}_precision".format(ds_type, val_i): all_corrects / count,
                        "epoch": epoch
                        })
                else:
                    wandb.log({
                        "valid_og_{}_precision".format(val_i): all_corrects / count,
                        "epoch": epoch
                        })

            if scheduler and val_i == apply_scheduler_at_val_set_i and epoch>start_scheduler_epoch:
                scheduler.step(all_corrects / count)

            results.append(all_corrects / count)
    return results

def generate_cfg():
    parser = argparse.ArgumentParser(description="CLEVR training/eval script")
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
        "--clevrh-only",
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

    cfg = get_default_cfg(mode='clevr')

    # update config with configuration file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if(cfg.LOAD_PATH in ['',' ',False,None] and not cfg.LOAD): #if load not specified in config but specified in args, then update in cfg
        cfg.LOAD_PATH = args.load_path
    if(args.load_path not in ['', False,' '] and not cfg.LOAD):
        cfg.LOAD=True
    cfg.freeze()
    return cfg, args.test_only, args.clevrh_only


if __name__ == '__main__':
    cfg, test_only, clevrh_only = generate_cfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print("using {}".format(device))
    
    save_model_every_eps = [20,25,30]

    
    if(not test_only):
        wandb.init(
            project=cfg.WANDB.PROJECT_NAME,
            name=cfg.WANDB.EXPERIMENT_NAME if cfg.WANDB.EXPERIMENT_NAME else None,
            config=config_to_wandb(cfg),
            tags=["IPRM"]
        )

    net = VLM_for_CLEVR(cfg).to(device)
    net_running = VLM_for_CLEVR(cfg).to(device)
    
    if cfg.LOAD:
        print("Loading model from {}".format(cfg.LOAD_PATH))
        with open(cfg.LOAD_PATH, 'rb') as f:
            state = torch.load(f, map_location=device)
        net.load_state_dict(state, strict=False)
    accumulate(net_running, net, 0)

    print(net)
    #print(f'num time steps: {net.vlm_module.max_computation_steps}')
    
          
    criterion = nn.CrossEntropyLoss()

    
    optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)

    # LR scheduler
    scheduler = None
    if cfg.SOLVER.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.SOLVER.FACTOR, patience=cfg.SOLVER.PATIENCE, threshold=0.001, threshold_mode='rel')

    start_scheduler_epoch = cfg.SOLVER.START_SCHEDULER_EPOCH
    #initialize dataloaders 
    train_dataloader, val_dataloaders = get_dataloader(cfg, mode='clevr', use_clevr_humans= False, 
                                                      clevr_train_split=cfg.CLEVR_TRAIN_SPLIT,
                                                      clevr_val_split=cfg.CLEVR_VAL_SPLIT,
                                                     )
    best_model_save_path = os.path.join(cfg.SAVE_DIRECTORY,'best_model.model')
    
    best_result = -1
    _, _, txt_processors = load_model_and_preprocess(name='blip_vqa', model_type='vqav2', is_eval=True, device='cpu') 
    del(_)
    
    if(not(os.path.exists(cfg.SAVE_DIRECTORY)) and not test_only):
        os.mkdir(cfg.SAVE_DIRECTORY)
    
    models_to_try_for_clevrh = {'clv_best': best_model_save_path}
    ds_type = 'clv_og'
    if(not clevrh_only):
        for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
            if(epoch==1 and not test_only):
                with open(os.path.join(cfg.SAVE_DIRECTORY, 'model_config.json'), 'w') as f:
                    json.dump(cfg, f, indent=4) 

            is_best=False
            if(not test_only):
                train(epoch, train_dataloader, txt_processors = txt_processors, ds_type = ds_type)

            results = valid(epoch, val_dataloaders, test_only=test_only, txt_processors = txt_processors, apply_scheduler_at_val_set_i=0, ds_type = ds_type, start_scheduler_epoch=start_scheduler_epoch)
            if(results[0]>best_result):
                best_result = results[0]
                is_best=True
                print(f"best result is now: {best_result}")
                

            if(test_only):
                break
            if(is_best):
                with open(best_model_save_path, 'wb') as f:
                    torch.save(net_running.state_dict(), f)
                
            if(epoch in save_model_every_eps):
                ep_model_save_path = os.path.join(cfg.SAVE_DIRECTORY,f'ep_{epoch}.model')
                with open(ep_model_save_path, 'wb') as f:
                    torch.save(net_running.state_dict(), f)
                models_to_try_for_clevrh[f'ep_{epoch}'] = ep_model_save_path

            
    if(clevrh_only):       
        models_to_try_for_clevrh = {}
        path_ = cfg.MODELS_FOR_CLEVR_H_ONLY_DIRECTORY
        for filen in os.listdir(path_):
            if(not filen.endswith('model')):
                continue
            models_to_try_for_clevrh[filen.split('.')[0]] = os.path.join(path_, filen) 
    #"""
    best_result = -1
    for m_key, model_start_path in models_to_try_for_clevrh.items():
        print("Now for CLEVR humans")
        print("Loading model from {}".format(model_start_path))
        with open(model_start_path, 'rb') as f:
            state = torch.load(f, map_location=device)
            net.load_state_dict(state, strict=False)
        #print(f'num time steps: {net.vlm_module.max_computation_steps}; Window len: {net.vlm_module.memory_window_len}')
    
        accumulate(net_running, net, 0)
        ds_type = f'clv_humn_{m_key}'
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)
        scheduler = None
        if cfg.SOLVER.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.SOLVER.FACTOR, patience=cfg.SOLVER.CLEVR_H_PATIENCE, threshold=0.001, threshold_mode='rel', verbose=True)

        train_dataloader, val_dataloaders = get_dataloader(cfg, mode='clevr', use_clevr_humans=True,
                                                          clevr_train_split=cfg.CLEVR_TRAIN_SPLIT,
                                                          clevr_val_split=cfg.CLEVR_VAL_SPLIT
                                                         )
        best_model_save_path = os.path.join(cfg.SAVE_DIRECTORY,f'best_clevr_humans_model_{m_key}.model')

        
        for epoch in range(1, cfg.SOLVER.CLEVR_HUMAN_EPOCHS + 1):
            if(epoch==1):
                results = valid(epoch, val_dataloaders, test_only=test_only, 
                            txt_processors = txt_processors, ds_type=ds_type+'_zero_shot')
            
            is_best=False
            if(not test_only):
                train(epoch, train_dataloader, txt_processors = txt_processors, ds_type=ds_type)

            results = valid(epoch, val_dataloaders, test_only=test_only, 
                            txt_processors = txt_processors, ds_type=ds_type, start_scheduler_epoch=start_scheduler_epoch)
            if(results[0]>best_result):
                best_result = results[0]
                is_best=True
                print(f"best result is now: {best_result}")
                
            if(test_only):
                break
            if(not(os.path.exists(cfg.SAVE_DIRECTORY))):
                os.mkdir(cfg.SAVE_DIRECTORY)

            if(is_best):
                with open(best_model_save_path, 'wb') as f:
                    torch.save(net_running.state_dict(), f)
