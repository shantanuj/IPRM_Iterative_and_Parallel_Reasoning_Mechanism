import os
import pickle
import numpy as np
import gensim
import torch

def convert_set_to_binary_feat(set_value_indices, max_value_len):
    #assume set_value_indices is zero indexed
    array = np.zeros(max_value_len, dtype=np.int64)
    array[list(set_value_indices)] = 1
    return array

def pad_operations_sequence(ops_seq, max_program_operations_len, end_of_compute_operation_token):
    current_ops_len = len(ops_seq)
    for i in range(max_program_operations_len-current_ops_len):
        ops_seq.append(end_of_compute_operation_token)
    #print(1213, ops_seq)
    return ops_seq

def pad_feat_values_sequence_numpy(feat_values_seq, max_program_operations_len, end_of_compute_feat_value_token, load_as_torch=False): 
    current_feat_values_len = len(feat_values_seq)
    max_feat_values = feat_values_seq[0].shape[0] #max_feat_values refers to the maximum possible values
    for i in range(max_program_operations_len-current_feat_values_len):
        binary_feat_for_end_of_compute = convert_set_to_binary_feat(set([end_of_compute_feat_value_token]), max_feat_values)
        feat_values_seq += [binary_feat_for_end_of_compute]
    if(load_as_torch):
        feat_values_seq = [torch.from_numpy(feat_val) for feat_val in feat_values_seq]
    return feat_values_seq

def pad_feat_values_sequence_set(feat_values_seq, max_program_operations_len, end_of_compute_feat_value_token): 
    current_feat_values_len = len(feat_values_seq)
    #max_feat_values = feat_values_seq[0].shape[0] #max_feat_values refers to the maximum possible values
    for i in range(max_program_operations_len-current_feat_values_len):
        feat_values_seq += [set([end_of_compute_feat_value_token])]
    #feat_values_seq = [torch.from_numpy(feat_val) for feat_val in feat_values_seq]
    return feat_values_seq
    
def collate_data(batch):
    images, lengths, answers, families, idxs, programs, ops_seqs, feat_values_seqs, classifier_types, pr_operations_lens= [], [], [], [], [], [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))
    #print(batch[0][-1], batch[0][-2], batch[0][-3])
    if(batch[0][-5] is not None):
        max_program_operations_len = max(map(lambda x: len(x[-5]), batch))
    #else:
        
    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    #operation_seqs_over_time = 
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family, idx, program, ops_seq, feat_values_seq, classifier_type, end_of_compute_operation_token, end_of_compute_feat_value_token, load_conjunct_feature_as_binary_numpy = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)
        idxs.append(idx)
        programs.append(program)
        
        #print(i,'-----------')
        if(ops_seq is not None):
            #print(11, ops_seq)
            program_ops_len = len(ops_seq)
            ops_seqs.append(pad_operations_sequence(ops_seq, max_program_operations_len, end_of_compute_operation_token))
            ##print(12, ops_seqs[-1])
            #print(21, feat_values_seq)
            if(load_conjunct_feature_as_binary_numpy):
                feat_values_seqs.append(pad_feat_values_sequence_numpy(feat_values_seq, max_program_operations_len, end_of_compute_feat_value_token))
            else:
                feat_values_seqs.append(pad_feat_values_sequence_set(feat_values_seq, max_program_operations_len, end_of_compute_feat_value_token))
            #print(22, feat_values_seqs[-1])
            #print(31, classifier_type)
            classifier_types.append(classifier_type)
            pr_operations_lens.append(program_ops_len)
            #print(32, classifier_types[-1])
        else:
            ops_seqs.append(0)                   
            feat_values_seqs.append(0)
            classifier_types.append(0)
            pr_operations_lens.append(0)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), families, idxs, programs, torch.LongTensor(ops_seqs), feat_values_seqs, torch.LongTensor(classifier_types), pr_operations_lens

def collate_data_GQA(batch):
    images, nums_objects, lengths, answers, idxs = [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[2]), batch))
    max_num_objects = max(map(lambda x: x[1], batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_q_len = sorted(batch, key=lambda x: len(x[2]), reverse=True)

    for i, b in enumerate(sort_by_q_len):
        image, _, question, length, answer, idx = b
        images.append(image[:,:,:max_num_objects])
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        idxs.append(idx)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), [], idxs

def load_pretrained_embedings(cfg, embeddings_table, device):
  # load pretrained embeddings from disc
  embeddings_w2v = gensim.models.KeyedVectors.load_word2vec_format(cfg.MAC.TRAINED_EMBD_PATH, binary=True)

  # load vocabulary
  features_path = os.path.join(cfg.DATALOADER.FEATURES_PATH, 'features')
  with open('{}/all_dic.pkl'.format(features_path), 'rb') as f:
    vocabulary = pickle.load(f)['word_dic']

  # overwrite default weights
  embedding_weights = embeddings_table.weight.detach()
  for key in vocabulary:
      if key in embeddings_w2v:
          id = vocabulary[key]
          embedding_weights[id] = torch.from_numpy(embeddings_w2v[key]).to(device)
  embeddings_table.weight = torch.nn.Parameter(embedding_weights)
