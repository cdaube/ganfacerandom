# this must be run in ganfaceapp environment

import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="0"
hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    proj0257Dir = '/analyse/Project0257/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    proj0257Dir = '/analyse/Project0257/'

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/stylegan2-master/'))
sys.path.append(os.path.abspath(homeDir+'GANFaceGenerationApp/'))

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks

nParticipants = 10
nBlocks = 10

# prepare styleGAN2
network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
seeds = [1000]
truncation_psi = 0.5

_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
if truncation_psi is not None:
    Gs_kwargs.truncation_psi = truncation_psi

seed = 1
rnd = np.random.RandomState(seed)

def genObsImages(latentVector,outPath,noise_vars,Gs,Gs_kwargs):
    
    savePth = outPath
    print('Generating image ...')
    z = latentVector[np.newaxis,:]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
    PIL.Image.fromarray(images[0], 'RGB').save(savePth)

for ss in range(nParticipants):

    # load data
    tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    thsLatent = tmp['coef_sg']
    # define filename
    outPath = f"{proj0257Dir}studentProjects/florian/figures/weightsInPixelSpace/F{ss+1:02d}_average.png"
    # generate image
    genObsImages(np.mean(thsLatent,axis=1),outPath,noise_vars,Gs,Gs_kwargs)

    # also generate individual folds
    for bb in range(nBlocks):
        # define filename
        outPath = f"{proj0257Dir}studentProjects/florian/figures/weightsInPixelSpace/individualFolds/F{ss+1:02d}_{bb+1:02d}.png"
        # generate image
        genObsImages(thsLatent[:,bb],outPath,noise_vars,Gs,Gs_kwargs)



