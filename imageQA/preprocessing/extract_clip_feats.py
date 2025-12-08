from torch.utils.data import Dataset, DataLoader
import torch
import h5py
from transformers import CLIPModel, AutoProcessor


#from transforms import Scale
import sys
import os
from PIL import Image
from tqdm import tqdm

#python3 extract_clip_feats.py /data/dataset/CLEVR/CLEVR_v1.0/ /data/usrdata/shan/CLEVR_feats/ clip_vit_large_336_res_no_crop
class CLEVR(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.length = len(os.listdir(os.path.join(root,'images', split)))
        print(os.path.join(root,'images', split))

    def __getitem__(self, index):
        img = os.path.join(self.root, 'images',
                        self.split,
                        'CLEVR_{}_{}.png'.format(self.split,
                                            str(index).zfill(6)))
        img = Image.open(img).convert('RGB')
        return img

    def __len__(self):
        return self.length
    
def collate_fn(batch):
    """
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    """
    return batch





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataset(split):
    root = sys.argv[1]
    out_path = sys.argv[2]
    model_type = sys.argv[3]
    batch_size=400
    dataloader = DataLoader(CLEVR(root, split), batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=4)

    size = len(dataloader)
    print(size)
    print(split, 'total', size * batch_size)
    print(device)
    if(model_type in ['clip_vit_large_336_res_no_crop','clip_vit_large_336_res_center_crop']):
        image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        if(model_type not in ['clip_vit_large_336_res_center_crop']):
            image_processor.feature_extractor.do_center_crop=False
            """##NOTE: In source code (installed in python library 'transformers') of transformers.image_processing_clip change below (and make modifications in resize func to accept arguments of default_to_square) (there are multiple places where default_to_square=False; set it to default_square=default_to_square): 
#if do_resize:
 #           images = [self.resize(image=image, size=size, resample=resample, default_to_square=not do_center_crop) for image in images]

            
            """

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        clip_model.to(device)

        tokens = 577
        dim = 1024
    elif(model_type in ['clip-vit-large-patch14-no-center-crop']):
        image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        image_processor.feature_extractor.do_center_crop=False
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_model.to(device)
        tokens = 257
        dim = 1024
        
    else:
        print('model type {} not found'.format(model_type))
        sys.exit(0)
             
    features_path = os.path.join(out_path, 'features_{}'.format(model_type))

    f = h5py.File('{}/{}_features.hdf5'.format(features_path, split), 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, tokens, dim),
                            dtype='f4')
    

    with torch.no_grad():
        prev_num =0
        for i, image in tqdm(enumerate(dataloader)):
            image_input = image_processor(images=image, return_tensors="pt").to(device)
            #print(image_input['pixel_values'].shape)
            outputs =  clip_model.vision_model.forward(**image_input)
            features = outputs['last_hidden_state'].detach().cpu() #clip_model.visual_projection(outputs['last_hidden_state']).detach().cpu()
            batch_size = len(image)
            dset[prev_num : prev_num + batch_size] = features
            
            prev_num+=batch_size
            print('0', batch_size, prev_num)#,## prev_num + batch_size)
            #batch_size = len(image)
            #print(batch_size)
            #dset[i * batch_size:(i + 1) * batch_size] = features

    f.close()
    
if __name__ == '__main__':
    create_dataset('val')
    create_dataset('train')