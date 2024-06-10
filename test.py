import os
from PIL import Image
from itertools import product
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from model.utils.utils import get_config
from model.trainers.Trainer_StyleFlow_DL import Trainer,set_random_seed

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None, mode='paired'):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): 'train' or 'test' to select the phase.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'paired' or 'mixed' to specify the pairing mode.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.mode = mode
        
        self.source_dir = os.path.join(root_dir, f'{phase}A')
        self.style_dir = os.path.join(root_dir, f'{phase}B')
        
        self.source_images = sorted(os.listdir(self.source_dir))
        self.style_images = sorted(os.listdir(self.style_dir))
        
        if mode == 'paired':
            self.pairs = list(zip(self.source_images, self.style_images))
        elif mode == 'mixed':
            self.pairs = list(product(self.source_images, self.style_images))
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source_img_name, style_img_name = self.pairs[idx]
        
        source_img_path = os.path.join(self.source_dir, source_img_name)
        style_img_path = os.path.join(self.style_dir, style_img_name)
        
        source_image = Image.open(source_img_path).convert('RGB')
        style_image = Image.open(style_img_path).convert('RGB')
        
        if self.transform:
            source_image = self.transform(source_image)
            style_image = self.transform(style_image)
        
        return (source_image,style_image)

code_name = 'coral'

root_dir = f'/home/alfin/Documents/deep_learning/StyleFlow_DL/Dataset/Dataset_{code_name}'
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

test_dataset_mixed = ImagePairDataset(root_dir=root_dir, 
                                      phase='test', 
                                      transform=transform, 
                                      mode='mixed')

test_loader_mixed = DataLoader(test_dataset_mixed, batch_size=1, shuffle=False, num_workers=4)

cfg_path = f'//home/alfin/Documents/deep_learning/StyleFlow_DL/config/{code_name}.yaml'
keep_ratio = [0.1,0.3,0.5,0.7,0.9]

for ratio in keep_ratio:
    ratio_name = str(int(ratio*100))
    checkpoint_path = f'/home/alfin/Documents/deep_learning/StyleFlow_DL/output/{code_name}_{ratio_name}_10_2/model_save/500.ckpt.pth.tar'

    args = get_config(cfg_path)
    args['keep_ratio'] = ratio
    print('using ratio of',args['keep_ratio'])
    trainer = Trainer(args)
    trainer.load_model(checkpoint_path)

    for batch_id,(source_image,style_image) in tqdm(enumerate(test_loader_mixed)):
        trainer.test(batch_id, source_image, style_image)
















