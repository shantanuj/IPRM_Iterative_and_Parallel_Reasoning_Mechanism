import cv2
from PIL import Image
from torchvision import transforms
import json
import pycocotools.mask as mask
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import math
#import io
#import base64  # Import the base64 module
def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
        #exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_list = []
    # Loop through the frames and convert them to PIL images
    for frame_number in range(frame_count):
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_number + 1}")
            continue

        # Convert the OpenCV frame to a PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Append the PIL image to the list
        frame_list.append(frame_pil)

    cap.release()
    return frame_list

def get_iteratively_sampled_indices(num_frames_to_sample, num_given_frames, verbose=False, return_sorted=True):
#num_frames_to_sample = 93
#num_given_frames = 62
    final_sampled_indices = []
    sampled_sets_over_rounds = []
    given_indices = list(range(0,num_given_frames))
    remaining_indices = given_indices
    round_i = 0
    while(len(final_sampled_indices)<num_frames_to_sample and len(remaining_indices)>0):
        if(verbose):
            print(f"Round {round_i}; Remaining samples len: {len(remaining_indices)}; Num already sampled len: {len(final_sampled_indices)}")
        num_frames_left_to_sample = num_frames_to_sample -len(final_sampled_indices)
        sample_rate = max(1, math.ceil(len(remaining_indices)/num_frames_left_to_sample))
        new_sampled_indices = remaining_indices[::sample_rate]
        final_sampled_indices.extend(new_sampled_indices)
        sampled_sets_over_rounds.append(new_sampled_indices)
        remaining_indices = list(set(given_indices) - set(final_sampled_indices))
        if(verbose):
            print(f"Sampled {len(new_sampled_indices)} more indices with sample rate {sample_rate} ")
    if(return_sorted):
        final_sampled_indices = sorted(final_sampled_indices)
    return final_sampled_indices, sampled_sets_over_rounds
#x= list(range(0,num_given_frames))
#sample_rate = max(1, math.ceil(num_given_frames/num_frames_to_sample))
#sample_rate, len(x[::sample_rate]), x[::sample_rate]
#sampled_indices = x[::sample_rate]

def get_video_frame_size(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    else:
        # Get the frame resolution
        frame_width = int(cap.get(4))  # Width of the frames
        frame_height = int(cap.get(3))  # Height of the frames

        #print(f"Frame resolution: {frame_width}x{frame_height}")

        # Release the video capture object
        cap.release()
        return (frame_width, frame_height)
    
def get_glove_embeddings(glove_file = '/data/usrdata/shan/Word_embeddings//glove.6B.300d.txt'):
    # Initialize empty dictionaries
    word_embeddings_torch = {}

    # Load GloVe embeddings from the file
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')  # Convert the vector to a NumPy array

            # Store the embeddings in the dictionaries
            word_embeddings_torch[word] = torch.tensor(vector)
    return word_embeddings_torch