import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
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


    
def plot_img_atts_together(img, img_att_tokens, num_mem_tokens, alpha=0.7, normalize=True, num_cols=3):
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
    plt.show()

def plot_lang_atts_across_times_together(img_att_imgs_time_wise, text_atts_time_wise, alpha=0.7, normalize=True, figsize=(12,2)):
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
    plt.show()
    
def plot_img_atts_across_times_together(img_att_imgs_time_wise, text_atts_time_wise, alpha=0.7, normalize=True, figsize=(12,2)):
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
    plt.show()

#m=0
#plt.imshow(blended_image)
def process_attention_over_time(img, text_att_tokens, img_att_tokens, og_question_text_list, 
                                vis_iterative_steps, vis_parallel_ops):
    """
    Process attention weights over time steps for both text and image modalities.
    
    Args:
        img: The input image
        text_att_tokens: List of text attention tokens for each time step
        img_att_tokens: List of image attention tokens for each time step
        og_question_text_list: Original question text as list of tokens
        vis_iterative_steps: Number of iterative time steps
        vis_parallel_ops: Number of parallel operations
    
    Returns:
        dict: Dictionary containing:
            - 'text_atts_over_t': List of tuples (attention_weights, input_tokens, output_tokens)
            - 'text_atts_over_t_meaned': List of meaned text attention weights
            - 'text_atts_over_t_maxed': List of maxed text attention weights
            - 'img_atts_over_t': List of image attention maps
            - 'img_atts_over_t_meaned': List of meaned image attention maps
            - 'img_atts_over_t_maxed': List of maxed image attention maps
    """
    img_atts_over_t = []
    text_atts_over_t = []
    text_atts_over_t_maxed = []
    text_atts_over_t_meaned = []
    img_atts_over_t_maxed = []
    img_atts_over_t_meaned = []

    for t in range(vis_iterative_steps):
        input_tokens_text = ['Op{}'.format(i) for i in range(vis_parallel_ops)]
        inter_output_tokens_text = og_question_text_list
        output_tokens_text = []
        for tok in inter_output_tokens_text:
            if(tok.startswith('Ġ')):
                output_tokens_text.append(tok[1:])
            else:
                output_tokens_text.append(tok)
        attention_weights_text_before = np.array(text_att_tokens[t].transpose(0,1).cpu())
        attention_weights_text_before = attention_weights_text_before[:len(output_tokens_text), :]
        
        text_atts_over_t.append((attention_weights_text_before, input_tokens_text, output_tokens_text))
        text_atts_over_t_meaned.append((attention_weights_text_before.mean(axis=1, keepdims=True), ['Cuml_Op'], output_tokens_text))
        text_atts_over_t_maxed.append((attention_weights_text_before.max(axis=1, keepdims=True), ['Cuml_Op'], output_tokens_text))

        img_atts = get_img_atts(img, img_att_tokens[t], vis_parallel_ops)
        img_atts_over_t.append(img_atts)
        img_atts_over_t_meaned.append(np.array(img_atts).mean(axis=0, keepdims=True))
        img_atts_over_t_maxed.append(np.array(img_atts).max(axis=0, keepdims=True))

    return {
        'text_atts_over_t': text_atts_over_t,
        'text_atts_over_t_meaned': text_atts_over_t_meaned,
        'text_atts_over_t_maxed': text_atts_over_t_maxed,
        'img_atts_over_t': img_atts_over_t,
        'img_atts_over_t_meaned': img_atts_over_t_meaned,
        'img_atts_over_t_maxed': img_atts_over_t_maxed
    }

    