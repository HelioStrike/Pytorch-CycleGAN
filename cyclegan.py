import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


#Hyperparameters
n_epochs = 200
batch_size = 1
val_batch_size = 5
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
image_height = 200
image_width = 200
channels = 3
sample_interval = 100
checkpoint_interval = 1
n_residual_blocks = 9
dataset_name = "hz"
print_every_iter = 10
load_model = False
dataset_path = "../../../../Data/horse2zebra/horse2zebra"

#paths
images_path = "./images/%s" % dataset_name
saved_models_path = "./saved_models/%s" % dataset_name

#Make directories
os.makedirs(images_path, exist_ok=True)
os.makedirs(saved_models_path, exist_ok=True)

#Loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

#cuda
cuda, device = [True, torch.device('cuda:0')] if torch.cuda.is_available() else [False, torch.device("cpu")]

#patch size
patch_size = (1, image_height//2**4, image_height//2**4)

#networks
G_AB = GeneratorResNet(channels, channels)
G_BA = GeneratorResNet(channels, channels)
D_A = Discriminator(channels)
D_B = Discriminator(channels)

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

#if cuda, make em cuda
criterion_GAN = criterion_GAN.to(device)
criterion_cycle = criterion_cycle.to(device)
criterion_identity = criterion_identity.to(device)

G_AB = G_AB.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)
D_B = D_B.to(device)

#load model
if load_model:
    G_AB.load_state_dict(torch.load(images_path + "/G_AB_" + dataset_name))
    G_BA.load_state_dict(torch.load(images_path + "/G_BA_" + dataset_name))
    D_A.load_state_dict(torch.load(images_path + "/D_A_" + dataset_name))
    D_B.load_state_dict(torch.load(images_path + "/D_B_" + dataset_name))

lambda_cyc = 10
lambda_id = 5

#optimizers
optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)

#Buffers of previouslty generated output
fake_A_buffer = ReplayMemory()
fake_B_buffer = ReplayMemory()

#transforms
transforms_ = [ transforms.Resize(int(image_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((image_height, image_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset(dataset_path, transforms_=transforms_, unaligned=True), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(ImageDataset(dataset_path, transforms_=transforms_, unaligned=True, mode='val'), batch_size=val_batch_size, shuffle=True)

def sample_image(epoch, i):
    imgs = next(iter(val_dataloader))
    real_A = imgs['A'].to(device)
    fake_B = G_AB(real_A)
    real_B = imgs['B'].to(device)
    fake_A = G_BA(real_B)
    save_image(torch.cat((real_A, fake_B, real_B, fake_A), 0).detach(), os.path.join(images_path, dataset_name + "_" + str(epoch) + "_" + str(i) + ".jpg"), nrow=5, normalize=True)

valid = torch.Tensor(np.ones((1, *patch_size))).to(device)
fake = torch.Tensor(np.zeros((1, *patch_size))).to(device)
valid.requires_grad = False
fake.requires_grad = False

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):

        #real images of domains A and B
        real_A = torch.Tensor(batch['A']).to(device)
        real_B = torch.Tensor(batch['B']).to(device)

        if(real_A.size(1) != 3 or real_B.size(1) != 3):
            continue

        fake_B = G_AB(real_A).to(device)
        fake_A = G_BA(real_B).to(device)

        #valid and fake labels

        #-------------------------
        #  Training the generators
        #-------------------------

        #generator Identity loss
        id_loss_A = criterion_identity(G_BA(real_A), real_A)
        id_loss_B = criterion_identity(G_AB(real_B), real_B)
        id_loss = (id_loss_A + id_loss_B)/2

        #generator GAN loss
        gan_loss_AB = criterion_GAN(D_B(fake_B), valid)
        gan_loss_BA = criterion_GAN(D_A(fake_A), valid)
        gan_loss = (gan_loss_AB + gan_loss_BA)/2

        #generator cycle loss
        cycle_loss_A = criterion_cycle(G_BA(fake_B), real_A)
        cycle_loss_B = criterion_cycle(G_AB(fake_A), real_B)
        cycle_loss = (cycle_loss_A + cycle_loss_B)/2

        gen_loss = lambda_id*id_loss + gan_loss + lambda_cyc*cycle_loss

        optimizer_G.zero_grad()
        gen_loss.backward()
        optimizer_G.step()

        #----------------------------
        #  Training Discriminator A
        #----------------------------

        loss_real_D_A = criterion_GAN(D_A(real_A.detach()), valid)
        loss_fake_D_A = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real_D_A + loss_fake_D_A)/2

        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        loss_real_D_B = criterion_GAN(D_B(real_B.detach()), valid)
        loss_fake_D_B = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real_D_B + loss_fake_D_B)/2

        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if i%print_every_iter == 0:
            print("Id Loss:", id_loss.item(), "Cycle Loss:", cycle_loss.item(), "GAN Loss:", gan_loss.item(), "D_A", loss_D_A.item(), "D_B", loss_D_B.item())
            print("Epoch:", epoch, "Iter:", i)
            sample_image(epoch, i)

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (dataset_name, epoch))
            torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (dataset_name, epoch))
            torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (dataset_name, epoch))
            torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (dataset_name, epoch))
