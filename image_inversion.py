import numpy as np 
import matplotlib.pyplot as plt 
import math 
## ichao : replace this to the styleGAN you found
from stylegan_layers import  G_mapping,G_synthesis

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import models
## from perceptual_model import VGG16_for_Perceptual
import torch.optim as optim
from torchvision import transforms
from PIL import Image 
import PIL

## which device to use
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def get_mean_latent(g_mapping, device):
    mean_w = None
    for i in range(10):
        z_latent = 2 * torch.randn((1000, 512), device=device) - 1
        w_latent = g_mapping(z_latent)
        mean = torch.mean(w_latent, dim=0, keepdim=True)
        if mean_w == None:
            mean_w = mean
        else:
            mean_w += mean
    mean_w /= 10
    print(mean_w.shape)
    return mean_w


def image_reader(img_path,resize=None):
    with open(img_path,"rb") as f: 
        image=Image.open(f)
        image=image.convert("RGB")
    if resize!=None:
        image=image.resize((resize,resize))
    #image = image.rotate(90)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    print(image.shape)
    return image

class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=[2,4,14,21]):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features 
        self.slice0=torch.nn.Sequential()
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()
        self.slice4=torch.nn.Sequential()
        self.slice5=torch.nn.Sequential()
        self.slice6=torch.nn.Sequential()

        for x in range(n_layers[0]):#relu1_1
            self.slice0.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[0],n_layers[1]): #relu1_2
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[1],n_layers[2]): #relu3_2
            self.slice2.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[2],n_layers[3]):#relu4_2
            self.slice3.add_module(str(x),vgg_pretrained_features[x])
        #for x in range(n_layers[3],n_layers[4]):#relu4_2
        #    self.slice4.add_module(str(x),vgg_pretrained_features[x])
        #for x in range(n_layers[4],n_layers[5]):#relu4_2
        #    self.slice5.add_module(str(x),vgg_pretrained_features[x])
        #for x in range(n_layers[5],n_layers[6]):#relu4_2
        #    self.slice6.add_module(str(x),vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False

    def forward(self,x):
        h0=self.slice0(x)
        h1=self.slice1(h0)
        h2=self.slice2(h1)
        h3=self.slice3(h2)
        #h4=self.slice4(h3)
        #h5=self.slice5(h4)
        #h6=self.slice6(h5)

        return h0,h1,h2,h3

def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution',default=1024,type=int)
    parser.add_argument('--src_im',default="sample.png")
    parser.add_argument('--src_dir',default="source_image/")
    parser.add_argument('--save_dir', default="save_image/encode1")
    parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
    parser.add_argument('--w_iteration',default=1000,type=int)
    parser.add_argument('--n_iteration',default=1000,type=int)
    parser.add_argument('--loop_time', default=5,type=int)
    args=parser.parse_args()

    ## ichao : this is the generator part, you can replace here using the generator you found  
    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis(resolution=args.resolution))    
    ]))
    ## ichao : load the pretrained generator's weight
    g_all.load_state_dict(torch.load(args.weight_file, map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping, g_synthesis=g_all[0], g_all[1]
    ## ichao : end of generator part
    
    ## ichao : read the input image (size : 3x1024x1024)
    name=args.src_im.split(".")[0]
    img=image_reader(args.src_dir+args.src_im) #(1,3,1024,1024) -1~1
    img=img.to(device)

    MSE_Loss = nn.MSELoss(reduction="mean")

    img_p=img.clone() ## ichao : used for perceptual loss
    ## ichao : resize the image to put into VGG
    upsample2d = torch.nn.Upsample(scale_factor=256/args.resolution, mode='bilinear') 

    img_p = upsample2d(img_p)
    # [4,9,16,23]
    # [2,4,14,21]
    perceptual_net = VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)

    mean_w = get_mean_latent(g_mapping, device)
    ## ichao : initialize the latent code we want to optimize
    dlatent = mean_w
    dlatent = dlatent.requires_grad_()
    synth_img = g_synthesis(dlatent)
    #dlatent = -2.0 * torch.randn((1,18,512), device=device) + 1.0
    #dlatent = dlatent.requires_grad_()
    
    #optimizer = optim.Adam({dlatent}, lr=0.01)
    # Latent code optimization
    loop_iteration = args.w_iteration + args.n_iteration
    print("Start")
    for loop in range(args.loop_time):
        for m in g_synthesis.blocks.values():
            m.epi1.top_epi[0].noise.requires_grad = False
            m.epi2.top_epi[0].noise.requires_grad = False

        print("========Latent Code Optimization=========")
        optimizer=optim.Adam({dlatent},lr=0.01,betas=(0.9,0.999),eps=1e-8)
        loss_list=[]
        for i in range(args.w_iteration):
            optimizer.zero_grad()
            
            synth_img = g_synthesis(dlatent)
            synth_img = (synth_img + 1.0) / 2.0 # Why
            mse_loss , perceptual_loss = caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,1,1,upsample2d)
            # adjust ratio to control the gradient part.
            # atio = 0.8
            # loss = (1 - ratio) * mse_loss + ratio * perceptual_loss
            loss = mse_loss + perceptual_loss
            loss.backward()

            optimizer.step()

            loss_np=loss.detach().cpu().numpy()
            loss_p=perceptual_loss.detach().cpu().numpy()
            loss_m=mse_loss.detach().cpu().numpy()

            loss_list.append(loss_np)
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(loop*loop_iteration+i,loss_np,loss_m,loss_p))
            if i%10==0:
                save_image(synth_img.clamp(0,1),"{dir}/{number}.png".format(dir=args.save_dir, number=loop*loop_iteration+i))

                np.save("latent_W/{}.npy".format(name),dlatent.detach().cpu().numpy())
        # Noise optimization
        print("============Noise Optimization============")
        dlatent.requires_grad = False
        noises = []
        for i, m in enumerate(g_synthesis.blocks.values()):
            m.epi1.top_epi[0].noise.requires_grad = True
            noises.append(m.epi1.top_epi[0].noise)
            m.epi2.top_epi[0].noise.requires_grad = True
            noises.append(m.epi2.top_epi[0].noise)
        
        optimizer=optim.Adam(noises,lr=5,betas=(0.9,0.999),eps=1e-8)
        for i in range(args.n_iteration):
            optimizer.zero_grad()
            ## ichao : generate an image using the current latent code
            #dlatent_ex= g_mapping(dlatent)
            synth_img = g_synthesis(dlatent)
            synth_img = (synth_img + 1.0) / 2.0 # Why
            mse_loss , perceptual_loss = caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,0,1,upsample2d)
            # adjust ratio to control the gradient part.
            # atio = 0.8
            # loss = (1 - ratio) * mse_loss + ratio * perceptual_loss
            loss = mse_loss + perceptual_loss
            loss.backward()

            optimizer.step()

            loss_np=loss.detach().cpu().numpy()
            loss_p=perceptual_loss.detach().cpu().numpy()
            loss_m=mse_loss.detach().cpu().numpy()

            loss_list.append(loss_np)
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(loop*loop_iteration+args.w_iteration+i,loss_np,loss_m,loss_p))
            if i%10==0:
                save_image(synth_img.clamp(0,1),"{dir}/{number}.png".format(dir=args.save_dir, number=loop*loop_iteration+args.w_iteration+i))
                np.save("noise/{}.npy".format(name),np.array(noises))

def caluclate_loss(synth_img, img, perceptual_net, img_p, MSE_Loss, perceptual_mue, MSE_mue, upsample2d):
    #calculate MSE Loss
    mse_loss = MSE_Loss(synth_img,img) # (lamda_mse/N)*||G(w)-I||^2

    #calculate Perceptual Loss
    real_0,real_1,real_2,real_3 = perceptual_net(img_p)
    synth_p = upsample2d(synth_img) #(1,3,256,256)
    synth_0,synth_1,synth_2,synth_3 = perceptual_net(synth_p)

    perceptual_loss=0
    perceptual_loss+=MSE_Loss(synth_0,real_0)
    perceptual_loss+=MSE_Loss(synth_1,real_1)
    perceptual_loss+=MSE_Loss(synth_2,real_2)
    perceptual_loss+=MSE_Loss(synth_3,real_3)
    #perceptual_loss+=MSE_Loss(synth_4,real_4)
    #perceptual_loss+=MSE_Loss(synth_5,real_5)
    #perceptual_loss+=MSE_Loss(synth_6,real_6)
    return MSE_mue * mse_loss,perceptual_mue * perceptual_loss
    
if __name__ == "__main__":
    main()



