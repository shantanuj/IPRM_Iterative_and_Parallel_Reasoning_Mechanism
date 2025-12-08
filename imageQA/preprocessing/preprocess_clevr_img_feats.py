import h5py
import torch
from torchvision.models.resnet import ResNet, resnet101
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import sys
#sys.path.append("../")
from transforms import Scale
import os
from PIL import Image
from tqdm import tqdm

from torch.autograd import Variable

#python3 image_feature.py /data/dataset/CLEVR/CLEVR_v1/ /data/usrdata/shan/CLEVR_cogen_feats/
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    return x

res = 272 #224
feat_res = res//16
transform = transforms.Compose([
    Scale((res, res)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class CLEVR(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.length = len(os.listdir(os.path.join(root,
                                                'images', split)))

    def __getitem__(self, index):
        img = os.path.join(self.root, 'images',
                        self.split,
                        'CLEVR_{}_{}.png'.format(self.split,
                                            str(index).zfill(6)))
        img = Image.open(img).convert('RGB')
        return transform(img)

    def __len__(self):
        return self.length

batch_size = 50

resnet = resnet101(True).to(device)
resnet.eval()
resnet.forward = forward.__get__(resnet, ResNet)

def create_dataset(split):
    root = sys.argv[1] #root to CLEVR root folder
    out_path = sys.argv[2] #root to output folder
    dataloader = DataLoader(CLEVR(root, split), batch_size=batch_size,
                            num_workers=4)

    size = len(dataloader)

    print(split, 'total', size * batch_size)

    features_path = os.path.join(out_path, f'features_{res}x{res}')
    if (not os.path.exists(features_path)):
        os.makedirs(features_path)
    
    f = h5py.File(f'{features_path}/{split}_features.hdf5', 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, 1024, feat_res, feat_res), #1024, 14, 14),
                            dtype='f4')

    with torch.no_grad():
        prev_num = 0
        for i, image in tqdm(enumerate(dataloader), total=size, desc=f'Processing {split}'):
            image = image.to(device)
            features = resnet(image).detach().cpu().numpy()
            dset[prev_num : prev_num + batch_size] = features
            prev_num += batch_size
    f.close()

if __name__ == '__main__':
    create_dataset('val')
    create_dataset('train')
