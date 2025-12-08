import os
import pickle
import numpy as np
from PIL import Image
import h5py
from glob import glob
import torch
import msgpack
import json
import re

from collections import Counter
from torch.utils.data import Dataset, DataLoader

def get_gqa_dataloader(cfg, pin_memory=True):
    train_gqa = GQA(root = cfg.DATALOADER.FEATURES_PATH,
                    features_pathname = cfg.DATALOADER.FEATURES_PATHNAME,
                    debug = False,
                    split = 'train',
                    load_semantic=False,
                    load_bboxes = True,
                    data_fraction=1.0)
    
    train_dataloader = DataLoader(
        train_gqa, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_gqa.collate_data, drop_last=True, shuffle=True, pin_memory=pin_memory,
    )
    val_gqa = GQA(root = cfg.DATALOADER.FEATURES_PATH,
                  features_pathname = cfg.DATALOADER.FEATURES_PATHNAME,
                debug = False,
                split = 'test',
                load_semantic=True,
                load_bboxes = True,
                data_fraction=1.0)
    val_dataloader = DataLoader(
        val_gqa, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_gqa.collate_data, drop_last=False, shuffle=False, pin_memory=pin_memory,
    )
    return train_dataloader, [val_dataloader]
    
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
    


class GQA(Dataset):
    #Adapted from https://github.com/kakao/DAFT
    def __init__(self, root, debug=False, split="train", 
                 data_fraction=1.0,
                 load_semantic=False,
                 load_bboxes=True,
                 features_pathname = "features_vinvl",
                 vinv_vl_objlabel2id_path = "/data/usrdata/shan/GQA/vin_vl_obj_label2id.json"
                ):#, family_key ='type'):
        self.processor = Preprocesser()
        self.root = root
        self.split = split
        self.name = "gqa"
        assert(split in ['train','test'])
        #self.family_key = family_key
        with open(f"{self.root}/imgs_aux_info.pkl", "rb") as f:
            self.object_info = pickle.load(f) #json.load(f)
        self.load_semantic = load_semantic
        self.load_bboxes = load_bboxes
        if(load_semantic):
            aux_str = '_with_semantic'
        else:
            aux_str = ""

        self.objlabel2id = json.load(open(vinv_vl_objlabel2id_path, 'rb'))
        
        if split == "train":
            data = list()
            
            with open(f"{self.root}/balanced_train_data{aux_str}.json", "r") as f:
                data += json.load(f, encoding="utf8")["questions"]
            with open(f"{self.root}/balanced_val_data{aux_str}.json", "r") as f:
                data += json.load(f, encoding="utf8")["questions"]
            self.data = data
            partial = int(len(self.data) * data_fraction)
            self.data = self.data[:partial]
        else:
            with open(f"{self.root}/balanced_testdev_data{aux_str}.json", "r") as f:
                self.data = json.load(f, encoding="utf8")["questions"]

        with open(f"{self.root}/newqVocabFile.json", "r") as f:
            qdic = json.load(f)
            self.qdic = dict(
                {
                    "w2i": {v: i for i, v in enumerate(qdic)},
                    "i2w": {i: v for i, v in enumerate(qdic)},
                }
            )

        with open(f"{self.root}/newaVocabFile.json", "r") as f:
            adic = json.load(f)
            self.adic = dict(
                {
                    "w2i": {v: i for i, v in enumerate(adic)},
                    "i2w": {i: v for i, v in enumerate(adic)},
                }
            )
            
        self.features_pathname = features_pathname

        if debug:
            self.data = self.data[:1000]

        self.pos_weight = self.get_weight()

    def __len__(self):
        return len(self.data)

    def get_weight(self):
        answers = [
            self.adic["w2i"][data["answer"]]
            if data["answer"] in self.adic["w2i"]
            else 1
            for data in self.data
        ]
        answers_cnt = Counter(answers)
        pos_weights = list()
        for i in range(len(self.adic["w2i"])):
            pos = answers_cnt[i]
            neg = len(answers) - pos
            if pos != 0:
                pos_weights.append(neg / pos)
            else:
                pos_weights.append(0)
        return torch.tensor(pos_weights)

    def pad_object_labels(self, obj_labels, max_len=100, pad_token='<UNK_PAD>'):
        return obj_labels + [pad_token for i in range(max_len-len(obj_labels))]
    
        
    def __getitem__(self, index):
        data = self.data[index]
        imgid, question, answer = (data["imageId"], data["question"], data["answer"])
        #program = data["semantic"] if self.load_semantic else None
        #q_type = data['type']
        #q_group = data['group']
        #question_tokens = self.processor.tokenize(question)
        
        #question_tokens = [
        #    self.qdic["w2i"][t] if t in self.qdic["w2i"] else 1 for t in question_tokens
        #]
        #q_approx_len = len(question_tokens)
        answer = self.adic["w2i"][answer] if answer in self.adic["w2i"] else 1

        info = self.object_info[imgid]

        img_path = f"{self.root}/{self.features_pathname}/{imgid}_feature.pth"
        img = torch.load(img_path)
        if(self.load_bboxes):
            bbox_path = f"{self.root}/{self.features_pathname}/{imgid}_bbox.pth"
            bbox = torch.load(bbox_path)
        else:
            bbox = None
        img_width, img_height = info['size'][0], info['size'][1]#['width'], info['height']
        obj_labels_padded = self.pad_object_labels(info['obj_labels'])
        
        obj_labels = torch.tensor([self.objlabel2id[objl] for objl in obj_labels_padded])
        
        
        return img, bbox, question, answer, index, img_width, img_height, obj_labels # imgid, program, q_type, q_group, info, img_width, img_height, q_approx_len

    def collate_data(self, batch):
        images, bboxes, questions, answers, indices, obj_labels = list(), list(), list(), list(), list(), list()
        
        #imgids, programs, q_types, q_groups, img_infos = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
        img_ws, img_hs = list(), list()
        batch_size = len(batch)

        #max_len = max(map(lambda x: x[-1], batch))
        #print(batch)
        sort_by_len = sorted(batch, key=lambda x: len(x[2]), reverse=True)
        #print(sort_by_len)
        for i, b in enumerate(sort_by_len):
            image, bbox, question, answer, index, img_w, img_h, obj_label = b
            #imgid, program, q_type, q_group, img_info, img_w, img_h, q_approx_len = b
            images.append(image)
            bboxes.append(bbox)
            questions.append(question)
            answers.append(answer)
            indices.append(index)
            #imgids.append(imgid)
            #programs.append(program)
            #q_types.append(q_type)
            #q_groups.append(q_group)
            #img_infos.append(img_info)
            img_ws.append(img_w)
            img_hs.append(img_h)
            #print(obj_label)
            obj_labels.append(obj_label)
            
        if(self.load_bboxes):
            bboxes = torch.stack(bboxes)
        return (
            torch.stack(images),
            bboxes,
            questions,
            torch.LongTensor(answers),
            indices,
            #programs,
            #q_types,
            #q_groups,
            #imgids,
            #img_infos,
            torch.Tensor(img_ws),
            torch.Tensor(img_hs),
            torch.stack(obj_labels)
        )

    
class Preprocesser(object):
    # prelim sentence tokenizer
    allPunct = ["?", "!", "\\", "/", ")", "(", ".", ",", ";", ":"]
    fullPunct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
        "%",
        "^",
        "&",
        "*",
        "~",
        "#",
        "$",
    ]
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }
    nums = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = {"a": "", "an": "", "the": ""}
    allReplaceQ = {}
    for replace in [contractions, nums, articles]:  # ,
        allReplaceQ.update(replace)

    allReplaceA = {}
    for replace in [contractions, nums]:  # ,
        allReplaceA.update(replace)

    periodStrip = lambda self, s: re.sub(r"(?!<=\d)(\.)(?!\d)", " ", s)  # :,' ?
    collonStrip = lambda self, s: re.sub(
        r"(?!<=\d)(:)(?!\d)", " ", s
    )  # replace with " " or ""?
    commaNumStrip = lambda self, s: re.sub(r"(\d)(\,)(\d)", r"\1\3", s)

    vqaProcessText = lambda self, text, tokenize, question: self.processText(
        text,
        ignoredPunct=list(),
        keptPunct=list(),
        endPunct=list(),
        delimPunct=self.fullPunct,
        replacelistPost=self.allReplaceQ if question else self.allReplaceA,
        reClean=True,
        tokenize=tokenize,
    )

    def processText(
        self,
        text,
        ignoredPunct=["?", "!", "\\", "/", ")", "("],
        keptPunct=[".", ",", ";", ":"],
        endPunct=[">", "<", ":"],
        delimPunct=list(),
        delim=" ",
        clean=False,
        replacelistPre=dict(),
        replacelistPost=dict(),
        reClean=False,
        tokenize=True,
    ):

        if reClean:
            text = self.periodStrip(text)
            text = self.collonStrip(text)
            text = self.commaNumStrip(text)

        if clean:
            for word in replacelistPre:
                origText = text
                text = text.replace(word, replacelistPre[word])
                if origText != text:
                    print(origText)
                    print(text)
                    print("")

            for punct in endPunct:
                if text[-1] == punct:
                    print(text)
                    text = text[:-1]
                    print(text)
                    print("")

        for punct in keptPunct:
            text = text.replace(punct, delim + punct + delim)

        for punct in ignoredPunct:
            text = text.replace(punct, "")

        for punct in delimPunct:
            text = text.replace(punct, delim)

        text = text.lower()

        ret = text.split()  # delim

        ret = [replacelistPost.get(word, word) for word in ret]

        ret = [t for t in ret if t != ""]
        if not tokenize:
            ret = delim.join(ret)

        return ret

    def tokenize(self, questionStr):
        question = self.vqaProcessText(questionStr, True, True)
        return question
    
