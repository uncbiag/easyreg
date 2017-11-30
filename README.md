# registration_net
net for image registration

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