import numpy as np 
import matplotlib.pyplot as plt 
from stylegan_layers import  G_mapping,G_synthesis

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim as optim
from torchvision import transforms
from PIL import Image 
import PIL

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def main():
    G = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=1024))    
    ]))
    ## ichao : load the pretrained generator's weight
    G.load_state_dict(torch.load("weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", map_location=device))
    G.eval()
    G.to(device)
    g_mapping, g_synthesis = G[0], G[1]
    dlatent = torch.randn((1, 512),device=device)
    dlatent = g_mapping(dlatent)
    #dlatent = dlatent.expand(1, 18, 512)
    synth_img = g_synthesis(dlatent)
    synth_img = (synth_img + 1.0) / 2.0

    save_image(synth_img.clamp(0,1),"source_image/sample_rand.png")
    counter = 0
    for i, m in enumerate(g_synthesis.blocks.values()):
        counter += 2
        m.epi1.top_epi[0].noise.requires_grad = True
        m.epi2.top_epi[0].noise.requires_grad = True
        print(counter)
if __name__ == '__main__':
    main()