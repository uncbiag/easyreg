# registration_net
net for image registration
# paper underwriting:

Training Ensemble in Once: Adaptive Cascaded Nets for Patch Based 3D Image Segmentation

https://www.overleaf.com/15181855brynfgjjfnmk

One Stop Mapping: Joint Networks for Longitudinal Image Registration

https://v2.overleaf.com/18560374xmmwkmtqmpzw

# Under developing
* 11.30 update dataloader 
* 11.21 add recurrent structure, add itering mod
* 11.8 fix bugs, add displacemet field constrains
* 11.7 add testing, tensorboard, run through
* 11.7 add bilinear, run through the training pipline
* 11.6 add inter- intra sampling, add dataManager, add SimpleNet
* 11.5 add prepare_data, dataloader
* 11.4 add affine Generator, control pointers Generator


# How to run:
Go to lib,  run sh make_cuda.sh\
Go to pipLine, run pipline

# To Do
*  complete framework
*  add minist test
*  test spline
*  Implement B spline(optional)
*  recurrent net
*  test cycle GAN as symmetric register
*  test conditonal GAN as discriminator