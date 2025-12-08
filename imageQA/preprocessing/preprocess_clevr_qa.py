import os
import sys
import json
import pickle

import nltk
import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from transforms import Scale

#Run (to generate for both CLEVR and CLEVR humans together): python3 preprocess.py /data/dataset/CLEVR/CLEVR_v1.0/questions/ /data/usrdata/shan/CLEVR_feats/ /data/usrdata/shan/Word_embeddings/glove.6B.300d.txt 1 1 /data/dataset/CLEVR/CLEVR-Humans/

#Run (to generate only for original CLEVR): python3 preprocess.py /data/dataset/CLEVR/CLEVR_v1.0/questions/ /data/usrdata/shan/CLEVR_feats/ /data/usrdata/shan/Word_embeddings/glove.6B.300d.txt 0 0 False

#(See at bottom for args)

def load_glove_embeddings(embedding_file):
    embedding_matrix = {}
    print("Loading glove embeddings from: {}".format(embedding_file))
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_matrix[word] = vector
      
    return embedding_matrix

def store_embedding_matrix(vocab, embedding_matrix, output_file_path):
    print("Storing pruned glove embedding matrix to: {}".format(output_file_path))
    embedding_size = len(list(embedding_matrix.values())[0])
    matrix = np.zeros((len(vocab), embedding_size), dtype=np.float32)
    for word, i in vocab.items():
        if word in embedding_matrix:
            matrix[i] = embedding_matrix[word]
    np.save(output_file_path, matrix)
    print("Storing pruned glove embedding matrix (size: {}; vocab size:{}) to: {}".format(len(matrix), len(vocab), output_file_path))

def process_question(root, split, out_path=None, word_dic=None, answer_dic=None, for_clevr_humans=False, both_clevr_and_clevr_humans=False):
    if word_dic is None:
        word_dic = {'<UNK>':0}

    if answer_dic is None:
        answer_dic = {}
    
    if(out_path is None):
        out_path = root

    if(for_clevr_humans):
        file = 'CLEVR-Humans-{}.json'.format(split)
    else:
        file = 'CLEVR_{}_questions.json'.format(split)
    with open(os.path.join(root, file)) as f:
        data = json.load(f)

    result = []
    word_index = len(word_dic)
    answer_index = len(answer_dic)

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            word = word.lower()
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']
        answer_word = answer_word.lower()
        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        if(for_clevr_humans):
            result.append((question['image_filename'], question_token, answer,
                    0, None))
        else:
            result.append((question['image_filename'], question_token, answer,
                    question['question_family_index'], question['program']))

    #root = sys.argv[1]
    features_path = os.path.join(out_path, 'features')

    if(not(os.path.exists(features_path))):
        os.makedirs(features_path)
        
    if(both_clevr_and_clevr_humans):
        if(for_clevr_humans):
            out_file = '{}/clevr_humans_and_original_joined_CLEVRhumans_{}.pkl'.format(features_path, split)
        else:
            out_file = '{}/clevr_humans_and_original_joined_CLEVRoriginal_{}.pkl'.format(features_path, split)
    
    else:
        if(for_clevr_humans):
            out_file = '{}/clevr_humans_{}.pkl'.format(features_path, split)
        else:
            out_file = '{}/clevr_original_{}.pkl'.format(features_path, split)
    with open(out_file, 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = sys.argv[1] #root to CLEVR questions folder (with train and val)
    out_path = sys.argv[2]
    glove_embeds_path = sys.argv[3] # set to '0' to not use glove embeddings
    for_clevr_humans = int(sys.argv[4]) #set to 1 to process CLEVR-Humans only
    both_clevr_and_clevr_humans = int(sys.argv[5]) #set to 1 to process both CLEVR and CLEVR-Humans together
    clevr_humans_root = sys.argv[6]
    
    if(both_clevr_and_clevr_humans):
        print("Processing original CLEVR")
        word_dic, answer_dic = process_question(root, 'train', out_path = out_path, for_clevr_humans=False, both_clevr_and_clevr_humans=both_clevr_and_clevr_humans)
        process_question(root, 'val', out_path = out_path, word_dic = word_dic, answer_dic = answer_dic, for_clevr_humans=False, both_clevr_and_clevr_humans=both_clevr_and_clevr_humans)
        print("Processing CLEVR Humans")
        process_question(clevr_humans_root, 'train', out_path = out_path, word_dic = word_dic, answer_dic = answer_dic, for_clevr_humans=True, both_clevr_and_clevr_humans=both_clevr_and_clevr_humans)
        process_question(clevr_humans_root, 'val', out_path = out_path, word_dic = word_dic, answer_dic = answer_dic, for_clevr_humans=True, both_clevr_and_clevr_humans=both_clevr_and_clevr_humans)
    else:
        word_dic, answer_dic = process_question(root, 'train', out_path = out_path, for_clevr_humans=for_clevr_humans)
        process_question(root, 'val', out_path = out_path, word_dic = word_dic, answer_dic = answer_dic, for_clevr_humans=for_clevr_humans)
        
    if(out_path is None):
        out_path = root
    features_path = os.path.join(out_path, 'features')
    
    if(glove_embeds_path not in ['0','f','False']):
        glove_embeds = load_glove_embeddings(glove_embeds_path)
        if(both_clevr_and_clevr_humans):
            output_glove_path = '{}/clevr_humans_and_original_glembd.npy'.format(features_path)
        elif(for_clevr_humans):
            output_glove_path = '{}/clevr_humans_glembd.npy'.format(features_path)
        else:
            output_glove_path = '{}/clevr_original_glembd.npy'.format(features_path)
        store_embedding_matrix(word_dic, glove_embeds, output_glove_path)
    
    if(both_clevr_and_clevr_humans):
        out_vocab_path = '{}/clevr_humans_and_original_vocab.pkl'.format(features_path)
    elif(for_clevr_humans):
        out_vocab_path = '{}/clevr_humans_vocab.pkl'.format(features_path)
    else:
        out_vocab_path = '{}/clevr_original_vocab.pkl'.format(features_path)
        
    with open(out_vocab_path, 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
        
##python3 preprocess.py /data/dataset/CLEVR/CLEVR_v1.0/questions/ /data/usrdata/shan/CLEVR_feats/ /data/usrdata/shan/Word_embeddings/glove.6B.300d.txt 1 1 /data/dataset/CLEVR/CLEVR-Humans/