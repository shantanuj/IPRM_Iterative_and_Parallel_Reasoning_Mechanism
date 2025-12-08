import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm_model import VLM_for_CLEVR
from config import get_default_cfg
import os
from dataset import get_dataloader
from torch.utils.data import DataLoader
from PIL import Image
from ipdb import set_trace as st
from lavis.models import load_model_and_preprocess
#import matplotlib



cfg = get_default_cfg(mode='clevr')
#config_file = "./configs/00_m6_t9_DISTIL-roberta_full_272_feats_CLEVRH_Only.yaml" #rmam_m6_t6_red_2_wind_2.yaml"
config_file = "/home/sjaiswa3/0_iprm/IPRM/CLEVR_exps/configs/best_configs/00000_BEST_m6_t9_DISTIL-roberta_full_272_feats.yaml" #m6_t9_redr1_16x16.yaml"
if config_file:
    cfg.merge_from_file(config_file)

device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

print(device)
#x = torch.load(cfg.VLM.PRETRAINED_LANG_ENCODER_PATH, weights_only=False)

net = VLM_for_CLEVR(cfg).to(device)

with open("/data/user_data/sjaiswa3/iprm_ckpts/ckpts/best_clevr_humans_model_clv_best.model", 'rb') as f:
#with open("/data/usrdata/shan/IPRM_ablations/m6_t9_redr1_16_16_CLEVRH_only/best_clevr_humans_model_ep_30.model", 'rb') as f:
    state = torch.load(f, map_location=device)
net.load_state_dict(state, strict=False)

cfg.DATALOADER.EVAL_BATCH_SIZE=32
cfg.DATALOADER.BATCH_SIZE=192
train_dataloader, val_dataloaders = get_dataloader(cfg, mode='clevr', use_clevr_humans= True, #cfg.USE_CLEVR_HUMANS,
                                                      clevr_train_split=cfg.CLEVR_TRAIN_SPLIT,
                                                      clevr_val_split=cfg.CLEVR_VAL_SPLIT,
                                                   train_percent=1.0,#0.002,
                                                       #use_vlm_dataloader =False#cfg.USE_VLM_DATALOADER
                                                     )
#from lavis.models import load_model_and_preprocess
_, _, txt_processors = load_model_and_preprocess(name='blip_vqa', model_type='vqav2', is_eval=True, device='cpu') 
del(_)
#net.vlm_module.max_computation_steps = 6

dl = val_dataloaders[0]

original_data_record = dl.dataset.data#[qid]
import pickle
with open("/data/user_data/sjaiswa3/iprm_datasets/clevr_feats/features/clevr_humans_and_original_vocab.pkl",'rb') as f:
    vocab = pickle.load(f)
#vocab = net.tokenizer.ids_to_tokens

try:
    id2w = net.tokenizer.ids_to_tokens#{v:k for k, v in vocab['word_dic'].items()}
except:
    id2w = {v:k[1:] for k,v in net.tokenizer.get_vocab().items()}
aid2w = {v:k for k, v in vocab['answer_dic'].items()}

#net.vlm_module.max_computation_steps = 6

import torch
import torchvision.transforms as transforms
from PIL import Image

def get_heat_map(original_img, att_tokens, 
                 scale_img = True,
                 res = (224,224),
                 cmap_img_heatmap = 'viridis', #'gray' #viridis
                 normalize=False,
                 alpha=0.6
                ):
    
   # img = Image.open(f"/data/dataset/CLEVR/CLEVR_v1.0/images/val/{original_img_filename}")
   # img = img.convert('RGB')
    
    transform = transforms.Compose([
    transforms.Resize(res) if scale_img else nn.Identity(),  # Resize to 224x224
    transforms.ToTensor()            # Convert PIL image to PyTorch tensor
    ])
    
    # Apply the transformation to the image
    resized_image = transform(original_img)
    
        # Convert the PyTorch tensor back to a NumPy array
    resized_image_np = resized_image.numpy()

    # Transpose the dimensions from CxHxW to HxWxC (PyTorch to Matplotlib format)
    resized_image_np = resized_image_np.transpose((1, 2, 0))
    
    or_att_tokens_size = att_tokens.shape[0]
    or_spatial_res = int(math.sqrt(or_att_tokens_size))
    resized_att_map = att_tokens.reshape(or_spatial_res, or_spatial_res).detach().cpu().numpy()
    resized_attention_map = np.array(Image.fromarray(resized_att_map).resize(res, Image.BILINEAR))

    
    original_image_np = resized_image_np#np.array(resized_image_np)
    if(normalize):
        normalized_attention_map = (resized_attention_map - resized_attention_map.min()) / (resized_attention_map.max() - resized_attention_map.min())
    else:
        normalized_attention_map = resized_attention_map
    
    heatmap = plt.get_cmap(cmap_img_heatmap)(normalized_attention_map)[:,:,:3]  # Use a colormap for visualization


    #alpha=0.8
    blended_image = (alpha * heatmap + (1 - alpha) * original_image_np)#.astype(np.uint8)
    return blended_image, heatmap, original_image_np


#import numpy as np
#import matplotlib.pyplot as plt

def visualize_seq2seq_attention(input_tokens, output_tokens, attention_weights, title = 'Attention Map',
                               cmap = 'gray', #viridis,
                                transposed=False
                               ):
    #plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
    plt.yticks(range(len(output_tokens)), output_tokens)
    
    if(transposed):
        plt.ylabel('Input Tokens')
        plt.xlabel('Output Tokens')
    else:
        plt.xlabel('Input Tokens')
        plt.ylabel('Output Tokens')
    plt.title(title)
    
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
alpha=0.7
normalize=True
num_cols = 3


def get_img_atts(img, img_att_tokens, num_mem_tokens, alpha=0.7, normalize=True, num_cols=3):
    num_rows = (num_mem_tokens + 1) // num_cols

    # Create a figure and axes for the subplots
    #fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    # Flatten the axes array if needed
    #if num_rows > 1:
     #   axes = axes.flatten()

    # Loop through image paths and axes, then display images
    img_atts = []
    for m in range(num_mem_tokens):
        #ax = axes[m]
        blended_image, heatmap, original_image_np = get_heat_map(img, img_att_tokens[m], alpha=alpha, normalize=normalize)
        img_atts.append(blended_image)
    return img_atts
        #ax.imshow(blended_image)
        #ax.set_title(f"Op {m}; T: {t}")
        #ax.axis("off")  # Turn off axis labels

    # Hide any remaining empty subplots
    #for i in range(num_mem_tokens, num_rows * num_cols):
     #   fig.delaxes(axes[i])

    # Adjust layout
    #plt.tight_layout()

    # Show the plot
    #plt.show()

def plot_img_atts_across_times_together(img_att_imgs_time_wise, text_atts_time_wise, alpha=0.7, normalize=True, b_i=0):
    #num_rows = (num_mem_tokens + 1) // num_cols
    num_cols = num_time_steps = len(img_att_imgs_time_wise) 
    num_rows = num_parallel_ops = len(img_att_imgs_time_wise[0]) +1
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    # Flatten the axes array if needed
    if num_rows > 1:
        axes = axes.flatten()

    # Loop through image paths and axes, then display images
    for t_i in range(num_time_steps):
        ax = axes[t_i]
        ax.imshow(text_atts_time_wise[t_i], cmap='gray', aspect='auto', vmin=0, vmax=1)
        
    for op_i in range(num_parallel_ops):
        for t_i in range(num_time_steps):
            count = (op_i+1)*t_i
            ax = axes[count]
            blended_image, heatmap, original_image_np = get_heat_map(img, img_att_tokens[count], alpha=alpha, normalize=normalize)
            ax.imshow(blended_image)
            ax.set_title(f"Op {op_i}; T: {t_i}")
            ax.axis("off")  # Turn off axis labels

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"img_atts_across_times_together_{b_i}.png", dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()
    
def plot_img_atts_together(img, img_att_tokens, num_mem_tokens, alpha=0.7, normalize=True, num_cols=3, b_i=0):
    num_rows = (num_mem_tokens + 1) // num_cols

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    # Flatten the axes array if needed
    if num_rows > 1:
        axes = axes.flatten()

    # Loop through image paths and axes, then display images
    for m in range(num_mem_tokens):
        ax = axes[m]
        blended_image, heatmap, original_image_np = get_heat_map(img, img_att_tokens[m], alpha=alpha, normalize=normalize)

        ax.imshow(blended_image)
        ax.set_title(f"Op {m}; T: {t}")
        ax.axis("off")  # Turn off axis labels

    # Hide any remaining empty subplots
    for i in range(num_mem_tokens, num_rows * num_cols):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"img_atts_together_{b_i}.png", dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_lang_atts_across_times_together(img_att_imgs_time_wise, text_atts_time_wise, alpha=0.7, normalize=True, figsize=(12,2), b_i=0):
    #num_rows = (num_mem_tokens + 1) // num_cols
    num_cols = num_time_steps = len(img_att_imgs_time_wise) 
    num_rows = 0#num_parallel_ops = len(img_att_imgs_time_wise[0])
    num_rows+=1
    
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, )
    #plt.subplots_adjust(wspace=0.0, hspace=0.0)
    # Flatten the axes array if needed
    if num_rows > 1:
        axes = axes.flatten()
    #plt.show()
    count=0    
    
    # Loop through image paths and axes, then display images
    for t_i in range(num_time_steps):
        ax = axes[count]
        #ax.imshow(text_atts_time_wise[t_i], cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.imshow(text_atts_time_wise[count][0], cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(text_atts_time_wise[count][1])), text_atts_time_wise[count][1], rotation=45)
        if(t_i==0):
            ax.set_yticks(range(len(text_atts_time_wise[count][2])), text_atts_time_wise[count][2])
        else:
            ax.set_yticks([])
        #else:
         #   ax.axis('off')
        count+=1
    
    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    # Adjust layout
    #plt.tight_layout()

    # Show the plot
    plt.savefig(f"lang_atts_across_times_together_{b_i}.png", dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()
    
def plot_img_atts_across_times_together(img_att_imgs_time_wise, text_atts_time_wise, alpha=0.7, normalize=True, figsize=(12,2), b_i=0):
    #num_rows = (num_mem_tokens + 1) // num_cols
    num_cols = num_time_steps = len(img_att_imgs_time_wise) 
    num_rows = num_parallel_ops = len(img_att_imgs_time_wise[0])
    #num_rows+=1
    
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, )
    #plt.subplots_adjust(wspace=0.0, hspace=0.0)
    # Flatten the axes array if needed
    if num_rows > 1:
        axes = axes.flatten()
    #plt.show()
    count=0    
    
    
    t_i=0
    for op_i in range(num_parallel_ops):
        for t_i in range(num_time_steps):
            #count = (op_i+1)*t_i
            #print(op_i, t_i)
            ax = axes[count]
            #blended_image, heatmap, original_image_np = get_heat_map(img, img_att_tokens[count], alpha=alpha, normalize=normalize)
            ax.imshow(img_att_imgs_time_wise[t_i][op_i])
            #ax.set_title(f"Op {op_i}; T: {t_i}")
            ax.axis("off")  # Turn off axis labels
            count+=1
    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    # Adjust layout
    #plt.tight_layout()
    #fig.text(0.5, 0.04, 'X-axis Label', ha='center', va='center', fontsize=12)

    #plt.xlabel('Computation steps (T)')
    #plt.ylabel('Parallel operations (N_op)')
    # Show the plot
    plt.savefig(f"img_atts_across_times_together_{b_i}_{t_i}.png", dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()
#m=0
#plt.imshow(blended_image)

#st()
for val_i, val_dataloader in enumerate(val_dataloaders):
    dataloader = iter(val_dataloader)
    net.eval()
    pbar = dataloader
    with torch.no_grad():
        for i, item in enumerate(pbar):
            image, question, answer = item['images'], [txt_processors['eval'](qs) for qs in item['questions']], item['answer_ids']
            #question[145] = "What shape is the object behind the small matte object with the maximum occuring shape?"#"What color is the object behind the cube with the maximum occuring color?
            #question[145] = "What shape is the object to the left of the red object having the maximum occuring shape?"
            #question[1] = "What object is behind the large cyan object with the maximum occuring shape?"
            #question[2] = "What shape is the object closest to the gray object with the maximum occuring shape?"
            #question[3] = "What shape is the object left of the blue small object with the maximum occuring shape?"
            image = image.to(device)
            q_len = None
            output = net(image, question, q_len)
            if(i==10):
                break


b_i = 3
atts = net.vlm_module.attentions
#net.vlm_module.num_memory_tokens=6
#net.vlm_module.max_computation_steps =9
num_mem_tokens = net.vlm_module.num_memory_tokens
num_computation_steps = net.vlm_module.max_computation_steps
#atts.keys()

num_mem_tokens, num_computation_steps, atts.keys()

print("Number of memory tokens: ", num_mem_tokens)
print("Number of computation steps: ", num_computation_steps)
print("Attention keys: ", atts.keys())

import matplotlib.pyplot as plt
import math
import numpy as np

st()
pred, gt_ans, og_question = output.argmax(-1)[b_i], answer[b_i].argmax(), question[b_i]
qid = item['q_ids'][b_i]
img_file = original_data_record[qid]['image_filename']#[0]
img = Image.open(f"/data/user_data/sjaiswa3/iprm_datasets/clevr/CLEVR_v1.0/images//val/{img_file}")
img = img.convert('RGB')
img_att_tokens = atts['image'].squeeze(-1).transpose(0,1)[b_i]
img_att_tokens = F.softmax(img_att_tokens, dim=2)
text_att_tokens =  atts['text'].squeeze(-1).transpose(0,1)[b_i]
text_att_tokens = F.softmax(text_att_tokens, dim=2)
mem_att_tokens =  atts['inter_mem_op'].squeeze(-1).transpose(0,1)[b_i]


#print(pred, gt_ans, og_question)c
pred_text, gt_ans_text = aid2w[int(pred)], aid2w[int(gt_ans)]
og_question_text_list = net.tokenizer.tokenize(og_question, add_special_tokens=False, return_length=True, padding=True)
og_question_text = og_question#net.tokenizer.tokenize(, add_special_tokens=False, return_length=True, padding=True)#model.tokog_question#' '.join(og_question_text_list)
print(f"Question: {og_question_text}\nPrediction: {pred_text}\nGround Truth: {gt_ans_text}")
plt.axis('off')
#plt.xlabel(f"Question:{question}")
plt.imshow(img)
plt.savefig(f"img_{b_i}.png")

# Collect attention data over timesteps
vis_iterative_steps = num_computation_steps
vis_parallel_ops = num_mem_tokens

prev_text_att_wt = None
prev_vis_att_wt = None
img_atts_over_t = []
text_atts_over_t = []
#cumulate_over_ops = True
#cumulate_over_iters_and_ops = True
#cumulate_mode = 'max'
text_atts_over_t_maxed = []
text_atts_over_t_meaned = []
img_atts_over_t_maxed = []
img_atts_over_t_meaned = []

text_atts_over_t_cumulate_ops = []
for t in range(vis_iterative_steps):
    input_tokens_text = ['Op{}'.format(i) for i in range(vis_parallel_ops)]#og_question_text_list
    inter_output_tokens_text = og_question_text_list#og_question_text.split(' ')#og_question_text_list#
    output_tokens_text = []
    for tok in inter_output_tokens_text:
        if(tok.startswith('Ġ')):
            output_tokens_text.append(tok[1:])
        else:
            output_tokens_text.append(tok)
    attention_weights_text_before = np.array(text_att_tokens[t].cpu().transpose(0,1))
    attention_weights_text_before = attention_weights_text_before[:len(output_tokens_text), :]
    
    text_atts_over_t.append((attention_weights_text_before, input_tokens_text, output_tokens_text))
    text_atts_over_t_meaned.append((attention_weights_text_before.mean(axis=1, keepdims=True), ['Cuml_Op'], output_tokens_text))
    text_atts_over_t_maxed.append((attention_weights_text_before.max(axis=1, keepdims=True), ['Cuml_Op'], output_tokens_text))

    #
    #visualize_seq2seq_attention(input_tokens_text, output_tokens_text, attention_weights_text_before, transposed=True, title=f"Before mem interaction Lang Attention T:{t}")

    img_atts = get_img_atts(img, img_att_tokens[t], vis_parallel_ops)
    #plot_img_atts_together(img, img_att_tokens[t], num_parallel_ops)
    img_atts_over_t.append(img_atts)
    img_atts_over_t_meaned.append(np.array(img_atts).mean(axis=0, keepdims=True))
    img_atts_over_t_maxed.append(np.array(img_atts).max(axis=0, keepdims=True))


#if(cumulate_over_ops):
#[0]
plot_img_atts_across_times_together(img_atts_over_t, text_atts_over_t, figsize=(12,12), b_i=b_i)