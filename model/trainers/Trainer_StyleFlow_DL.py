import random
import numpy as np
import os

import torch
import torch.nn as nn
import model.network.net as net

from torchvision.utils import save_image
from model.network.glow import Glow
from model.utils.utils import IterLRScheduler,remove_prefix
from model.layers.activation_norm import calc_mean_std
from model.losses.tv_loss import TVLoss

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename+'.pth.tar')

class merge_model(nn.Module):
    def __init__(self,cfg):
        super(merge_model,self).__init__()
        self.glow = Glow(3, cfg['n_flow'], cfg['n_block'], affine=cfg['affine'], conv_lu=not cfg['no_lu'])

    def forward(self,content_images, domain_class):
        z_c = self.glow(content_images, forward=True)
        stylized = self.glow(z_c, forward=False, style=domain_class)

        return stylized

def get_smooth(I, direction):
        weights = torch.tensor([[0., 0.],
                                [-1., 1.]]
                                ).cuda()
        weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
        weights_y = torch.transpose(weights_x, 0, 1)
        if direction == 'x':
            weights = weights_x
        elif direction == 'y':
            weights = weights_y

        F = torch.nn.functional
        output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))  # stride, padding
        return output

def avg(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    gradients_I_x = get_smooth(I_gray,'x')
    gradients_I_y = get_smooth(I_gray,'y')

    return torch.mean(gradients_I_x * torch.exp(-10 * avg(R_gray, 'x')) + gradients_I_y * torch.exp(-10 * avg(R_gray, 'y')))
    
class Trainer():
    def __init__(self,cfg,seed=0):
        self.init = True
        set_random_seed(seed)
        self.cfg = cfg
        Mmodel = merge_model(cfg)
        
        self.model = Mmodel
        self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.lr_scheduler = IterLRScheduler(self.optimizer, cfg['lr_steps'], cfg['lr_mults'], last_iter=cfg['last_iter'])
        
        vgg = net.vgg
        vgg.load_state_dict(torch.load(cfg['vgg']))
        self.encoder = net.Net(vgg,cfg['keep_ratio']).cuda()
      

        self.tvloss = TVLoss().cuda()

        self.model_log_path = os.path.join(self.cfg['output'], 
                                          '{}_{}_{}_{}'.format(self.cfg['job_name'],
                                                            str(int(self.cfg['keep_ratio']*100)),
                                                            str(self.cfg['n_flow']),
                                                            str(self.cfg['n_block'])), 
                                          'model_save')

        self.img_log_path = os.path.join(self.cfg['output'], 
                                          '{}_{}_{}_{}'.format(self.cfg['job_name'],
                                                            str(int(self.cfg['keep_ratio']*100)),
                                                            str(self.cfg['n_flow']),
                                                            str(self.cfg['n_block'])), 
                                          'img_save')

        self.img_test_path = os.path.join(self.cfg['output'], 
                                          '{}_{}_{}_{}'.format(self.cfg['job_name'],
                                                            str(int(self.cfg['keep_ratio']*100)),
                                                            str(self.cfg['n_flow']),
                                                            str(self.cfg['n_block'])), 
                                          'test')

        if not os.path.exists(self.model_log_path):
          os.makedirs(self.model_log_path)

        if not os.path.exists(self.img_log_path):
          os.makedirs(self.img_log_path)

        if not os.path.exists(self.img_test_path):
          os.makedirs(self.img_test_path)

    def load_model(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self,batch_id, content_iter, style_iter):
        content_images = content_iter.cuda()
        style_images = style_iter.cuda()
        target_style = style_iter
        
        domain_weight = torch.tensor(1).cuda()

        if self.init:
            base_code = self.encoder.cat_tensor(style_images.cuda())
            self.model(content_images,domain_class=base_code.cuda())
            self.init = False
            return

        base_code = self.encoder.cat_tensor(target_style.cuda())
        stylized = self.model(content_images,domain_class=base_code.cuda())
        stylized = torch.clamp(stylized,0,1)

        if self.cfg['loss'] == 'tv_loss':
            smooth_loss = self.tvloss(stylized)
        else:
            smooth_loss = get_gradients_loss(stylized, target_style.cuda())
        

        loss_c, loss_s = self.encoder(content_images, style_images, stylized, domain_weight)
        loss_c = loss_c.mean().cuda()
        loss_s = loss_s.mean().cuda()

        Loss = self.cfg['content_weight']*loss_c + self.cfg['style_weight']*loss_s + smooth_loss

        Loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        reduce_loss = Loss.clone()
        loss_c_ = loss_c.clone()
        loss_s_ = loss_s.clone()
        smooth_loss_ = smooth_loss.clone()

        current_lr = self.lr_scheduler.get_lr()[0]

        if batch_id % 100 == 0:
            output_name = os.path.join(self.img_log_path,str(batch_id)+'.jpg')
            # output_images = torch.cat((content_images.cpu(), stylized.cpu() , style_images.cpu()), 
            #                         0)
            print('saved at ',output_name)

            content_image = content_images[0].unsqueeze(0)
            stylized_image = stylized[0].unsqueeze(0)
            style_image = style_images[0].unsqueeze(0)

            output_images = torch.cat((content_image, stylized_image, style_image), dim=3)
            save_image(output_images, output_name, nrow=1)


        if batch_id % 500 == 0:
            save_checkpoint({
                'step':batch_id,
                'state_dict':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict()
                },os.path.join(self.model_log_path,str(batch_id)+ '.ckpt'))

    def test(self,batch_id, content_iter, style_iter):
        content_images = content_iter.cuda()
        style_images = style_iter.cuda()
        target_style = style_iter
        
        domain_weight = torch.tensor(1).cuda()

        if self.init:
            base_code = self.encoder.cat_tensor(style_images.cuda())
            self.model(content_images,domain_class=base_code.cuda())
            self.init = False
            return

        with torch.no_grad():
            base_code = self.encoder.cat_tensor(target_style.cuda())
            stylized = self.model(content_images,domain_class=base_code.cuda())
            stylized = torch.clamp(stylized,0,1)

        for img_idx in range(len(content_images)):
            output_name_ori = os.path.join(self.img_test_path,str(batch_id)+f'_ori_{img_idx}.jpg')
            output_name_target = os.path.join(self.img_test_path,str(batch_id)+f'_target_{img_idx}.jpg')
            output_name_style = os.path.join(self.img_test_path,str(batch_id)+f'_style_{img_idx}.jpg')

            content_image = content_images[img_idx].unsqueeze(0)
            stylized_image = stylized[img_idx].unsqueeze(0)
            style_image = style_images[img_idx].unsqueeze(0)

            # output_images = torch.cat((content_image, stylized_image, style_image), dim=3)
            save_image(content_image, output_name_ori, nrow=1)
            save_image(stylized_image, output_name_target, nrow=1)
            save_image(style_image, output_name_style, nrow=1)
            print('saved at ',output_name_ori)