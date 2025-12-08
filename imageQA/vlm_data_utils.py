import torch
import os
import json, pickle
from PIL import Image

import h5py

from torch.utils.data import DataLoader
from ipdb import set_trace as st

class CLEVR_for_VLM_dataloader(torch.utils.data.Dataset):
    def __init__(self, 
                 root, 
                 split,#='train', 
                 model_config=None,
                 images_root_path = "/data/dataset/CLEVR/CLEVR_v1.0/images/",
                 
                 make_new_label_config=True,
                 answer_vocab_path = 'clevr_labels_tuple.pkl',
                 load_img_feats=False,
                 img_feats_path=None,#os.path.join(img_feats_path,'train_features.hdf5')
                 train_percent=1.0, 
                 
                 convert_RGB=True,
                 #return_answer_as_text=False,
                 #use_clevr_humans='original_clevr_only', 
                 is_clevr_humans=False,
                 clevr_closure_question_type = None,
                 #clevr_closure_set_type=None
                ):
        
        #features_path = os.path.join(root, 'features')
        #with open('{}/clevr_original_{}.pkl'.format(features_path, split), 'rb') as f:
         #   data = pickle.load(f)
        if(is_clevr_humans):
            with open(os.path.join(root, "CLEVR-Humans-{}.json".format(split)), 'rb') as f:
                self.data = json.load(f)['questions']
        elif(clevr_closure_question_type is not None):
            with open(os.path.join('./Closure/', f"{clevr_closure_question_type}_{split}.json"), 'rb') as f:
                self.data = json.load(f)['questions']
            
            
            
        else:
            with open(os.path.join(root,  "CLEVR_{}_questions.json".format(split)), 'rb') as f:
                self.data = json.load(f)['questions']
        if(images_root_path is not None):
            self.images_root_path = os.path.join(images_root_path, split)
        
        self.convert_RGB = convert_RGB
        self.split = split
        self.is_clevr_humans = is_clevr_humans
        self.make_new_label_config = make_new_label_config
        
        #    self.data = json.load(os.patht.join(root,,'rb') as f:
        if(self.make_new_label_config):
            print('answer vocab path: ', answer_vocab_path)
            if(not os.path.exists(answer_vocab_path)):
                print("WARNING: Creating new answr vocab path")
                self._create_answer_vocab(answer_vocab_path)
            
            with open(answer_vocab_path,'rb') as f:
                self.clevr_label2id, self.clevr_id2label, self.clevr_labels_len = pickle.load(f)
        else:
            self.clevr_label2id, \
            self.clevr_id2label, \
            self.clevr_labels_len = model_config.label2id, model_config.id2label, len(model_config.id2label)
            not_found_ans_labels_list, num_qs_with_not_found_ans = self.get_not_found_answer_labels(self.data, model_config)
            print("Not found answer labels in original model config: {}; Num qs where this occurs: {} out of {}. ({:.2f}%)".format(not_found_ans_labels_list, num_qs_with_not_found_ans, len(self.data), num_qs_with_not_found_ans/len(self.data)*100))
            if(len(not_found_ans_labels_list)>0):
                print("Removing qs without matching answer labels")
                self.data, num_qs_removed = self.filter_questions_with_ignore_output_labels(self.data, not_found_ans_labels_list)
                assert(num_qs_removed == num_qs_with_not_found_ans)
                print("Done")
                  
            #Find 
        self.train_percent = train_percent
        if(is_clevr_humans):
            self.program_lens_dict, self.program_types_dict = {}, {}
        else:
            self.program_lens_dict, self.program_types_dict = self.get_program_info(self.data)
        
        self.load_image_feats = load_img_feats
        if(load_img_feats):
            self.h = h5py.File(img_feats_path, 'r')
            self.imgs = self.h['data']

    def close(self):
        self.h.close()
        
    #pickle.dump((clevr_label2id, clevr_id2label, len(clevr_label2id)),f)
    
    def get_program_type(self, program):
        if(program is None):
            return 'n/a'
        program_type = program[-1]['function'].split('_')[0] #last step
        if(program_type in ['greater', 'less','lesser']):
            program_type = 'comparitive_num'
        if(program[-1]['function'] == 'equal_integer'):
            program_type = 'equal_num'
        return program_type
        
    def get_program_info(self, data):
        program_lens_dict = {}
        program_types_dict = {}
        for item in data:
            program = item['program']
            if(len(program) not in program_lens_dict):
                program_lens_dict[len(program)] = 0
            program_lens_dict[len(program)] += 1
            
            pr_type = self.get_program_type(program)
            if(pr_type not in program_types_dict):
                program_types_dict[pr_type] = 0
            program_types_dict[pr_type] += 1
        return program_lens_dict, program_types_dict
                
        
    def get_not_found_answer_labels(self, original_qs, model_config):
        not_found_answers = set()
        normal_answers = set()
        num_qs_with_not_found_ans =0
        for item in original_qs:
            answer = item['answer']
            normal_answers.add(answer)
            if(answer not in model_config.label2id):
                not_found_answers.add(answer)
                num_qs_with_not_found_ans+=1
        return list(not_found_answers), num_qs_with_not_found_ans
    
    def filter_questions_with_ignore_output_labels(self, 
                                                   original_questions, 
                                                   ignore_outputs_list, #= ['cube', 'cyan', 'sphere']
                                                   ):
        new_questions = []
        num_original_qs = len(original_questions)

        for item in original_questions:
            answer = item['answer']
            if(answer not in ignore_outputs_list):
                new_questions.append(item)
        num_new_qs = len(new_questions)
        return new_questions, num_original_qs - num_new_qs

    def _create_answer_vocab(self, answer_vocab_path):
        clevr_label2id = dict()
        for item in self.data:
            answer = item['answer'].strip().lower()
            if(answer not in clevr_label2id):
                clevr_label2id[answer] = len(clevr_label2id)
            
        clevr_id2label = {v:k for k,v in clevr_label2id.items()}
        with open(answer_vocab_path,'wb') as f:
            pickle.dump((clevr_label2id, clevr_id2label, len(clevr_label2id)),f)
        
    def load_clevr_image(self,
                         file_name, 
                         images_root_path,
                         convert_RGB=False
                        ):
        img_path = os.path.join(self.images_root_path, file_name)
        image = Image.open(img_path)
        if(convert_RGB):
            image = image.convert('RGB')
        return image

    

    def __getitem__(self, index):
        item = self.data[int(index/self.train_percent)]
        #img_id = item['
        question = item['question']
        answer_text = item['answer']
        #st()
        #print("clevr label2id: ", self.clevr_label2id)
        answer_id = self.clevr_label2id[answer_text]
        target = torch.zeros(self.clevr_labels_len)
        target[answer_id] = 1
        image_filename = item['image_filename']
        if(self.load_image_feats):
            id = int(image_filename.rsplit('_', 1)[1][:-4])
            image = torch.from_numpy(self.imgs[id])
        else:
            image = self.load_clevr_image(image_filename,
                                      self.images_root_path,
                                      self.convert_RGB
                                     )
        if(self.is_clevr_humans):
            program = None
            question_family_index = None
        else:
            program = item['program']
            question_family_index = item['question_family_index']
        
        return question, answer_text, target, image, program, question_family_index, int(index/self.train_percent)
    
    def __len__(self):
        return int(len(self.data) * self.train_percent)
    

def collate_fn_vlm(batch):
    """
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    """
    sort_by_len = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    batch = {'questions':[],
             'answer_texts':[],
             'answer_ids':[],
             'images':[],
             'programs':[],
             'q_families':[],
             'q_ids':[],
            }
    
    for i, b in enumerate(sort_by_len):
        question, answer_text, answer_id, image, program, question_family_index, q_id = b
        batch['questions'].append(question)
        batch['answer_texts'].append(answer_text)
        batch['answer_ids'].append(answer_id)
        batch['images'].append(image)
        batch['programs'].append(program)
        batch['q_families'].append(question_family_index)
        batch['q_ids'].append(q_id)
        
    #print(batch['answer_ids'])
    batch['answer_ids'] = torch.stack(batch['answer_ids'])
    if(type(batch['images'][0]) == torch.Tensor):
        batch['images'] = torch.stack(batch['images'])
    #print(type(batch['images'][0]))
    #input()
          
    # create padded pixel values and corresponding pixel mask
    #encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    return batch


class GQA_for_VILT(torch.utils.data.Dataset):
    def __init__(self, 
                 root, 
                 split,#='train', 
                 make_new_label_config,
                 model_config=None,
                 #processor,
                 
                 answer_vocab_path = 'gqa_labels_tuple.pkl',
                 images_root_path = "/data/usrdata/shan/GQA/images/",
                 
                 train_percent=1.0, 
                 
                 convert_RGB=False, #as already jpg 
                 #return_answer_as_text=False,
                 #use_clevr_humans='original_clevr_only', 
                 ):
        
        #features_path = os.path.join(root, 'features')
        #with open('{}/clevr_original_{}.pkl'.format(features_path, split), 'rb') as f:
         #   data = pickle.load(f)
            #balanced_trainvalcombined_data_with_semantic
        with open(os.path.join(root,  "balanced_{}_data_with_semantic.json".format(split)), 'r') as f:
            self.data = json.load(f)['questions']
        self.images_root_path = images_root_path#os.path.join(images_root_path, split)
        self.convert_RGB = convert_RGB
        self.split = split
        self.make_new_label_config = make_new_label_config
        
        #    self.data = json.load(os.patht.join(root,,'rb') as f:
        if(self.make_new_label_config):
            if(not os.path.exists(answer_vocab_path)):
                print("WARNING: Creating new answr vocab path at: {}".format(answer_vocab_path))
                self._create_answer_vocab(answer_vocab_path)
            
            with open(answer_vocab_path,'rb') as f:
                print("Loading answer vocab from: ", answer_vocab_path)
                self.label2id, self.id2label, self.labels_len, self.label2freq = pickle.load(f)
        
        else:
            self.label2id, \
            self.id2label, \
            self.labels_len = model_config.label2id, model_config.id2label, len(model_config.id2label)
            self.label2freq = None
            not_found_ans_labels_list, num_qs_with_not_found_ans = self.get_not_found_answer_labels(self.data, model_config)
            print("Not found answer labels in original model config: {}; Num qs where this occurs: {} out of {}. ({:.2f}%)".format(not_found_ans_labels_list, num_qs_with_not_found_ans, len(self.data), num_qs_with_not_found_ans/len(self.data)*100))
            if(len(not_found_ans_labels_list)>0):
                print("Removing qs without matching answer labels")
                self.data, num_qs_removed = self.filter_questions_with_ignore_output_labels(self.data, not_found_ans_labels_list)
                assert(num_qs_removed == num_qs_with_not_found_ans)
                print("Done")
                  
            #Find 
        self.train_percent = train_percent
        
        self.program_lens_dict, self.program_types_dict = self.get_program_info(self.data)
    #pickle.dump((clevr_label2id, clevr_id2label, len(clevr_label2id)),f)
    
    """
    def get_program_type(self, program):
        program_type = program[-1]['function'].split('_')[0] #last step
        if(program_type in ['greater', 'less','lesser']):
            program_type = 'comparitive_num'
        if(program[-1]['function'] == 'equal_integer'):
            program_type = 'equal_num'
        return program_type
    """    
    def get_program_info(self, data):
        program_lens_dict = {}
        program_types_dict = {}
        for item in data:
            program = item['semantic']
            if(len(program) not in program_lens_dict):
                program_lens_dict[len(program)] = 0
            program_lens_dict[len(program)] += 1
            
            pr_type = item['type'] #self.get_program_type(program)
            if(pr_type not in program_types_dict):
                program_types_dict[pr_type] = 0
            program_types_dict[pr_type] += 1
        return program_lens_dict, program_types_dict
        
    def get_not_found_answer_labels(self, original_qs, model_config):
        not_found_answers = set()
        normal_answers = set()
        num_qs_with_not_found_ans =0
        for item in original_qs:
            answer = item['answer'].strip().lower()
            normal_answers.add(answer)
            if(answer not in model_config.label2id):
                not_found_answers.add(answer)
                num_qs_with_not_found_ans+=1
        return list(not_found_answers), num_qs_with_not_found_ans
    
    def filter_questions_with_ignore_output_labels(self, 
                                                   original_questions, 
                                                   ignore_outputs_list, #= ['cube', 'cyan', 'sphere']
                                                   ):
        new_questions = []
        num_original_qs = len(original_questions)

        for item in original_questions:
            answer = item['answer'].strip().lower()
            if(answer not in ignore_outputs_list):
                new_questions.append(item)
        num_new_qs = len(new_questions)
        return new_questions, num_original_qs - num_new_qs

    def _create_answer_vocab(self, answer_vocab_path):
        label2id = dict()
        label2freq = dict()
        label2id['<UNK>'] = 0
        label2freq['<UNK>'] = 0
        
        for item in self.data:
            answer = item['answer'].strip().lower()
            if(answer not in label2id):
                label2id[answer] = len(label2id)
                label2freq[answer] = 0
            label2freq[answer]+=1
        id2label = {v:k for k,v in label2id.items()}
        #print(label2id)
        with open(answer_vocab_path,'wb') as f:
            pickle.dump((label2id, id2label, len(id2label), label2freq),f)
        
    def load_image(self,
                         file_name, 
                         images_root_path,
                         convert_RGB=False
                        ):
        img_path = os.path.join(self.images_root_path, file_name)
        image = Image.open(img_path)
        if(convert_RGB):
            image = image.convert('RGB')
        return image

    

    def __getitem__(self, index):
        item = self.data[int(index/self.train_percent)]
        question = item['question']
        answer_text = item['answer'].strip().lower()
        if(answer_text in self.label2id):
            answer_id = self.label2id[answer_text]
        else:
            print("Warning: Label {} not found in label2id.. settting to <UNK>".format(answer_text))
            answer_id = self.label2id['<UNK>']
            
        target = torch.zeros(self.labels_len)
        target[answer_id] = 1
        image_filename = item['imageId'] + ".jpg"
        image = self.load_image(image_filename,
                                      self.images_root_path,
                                      self.convert_RGB
                                     )
        program = item['semantic']
        question_family_index = item['type']
        
        return question, answer_text, target, image, program, question_family_index, index 
    
    def __len__(self):
        return int(len(self.data) * self.train_percent)