from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from .modules import *
class MomentumNet(nn.Module):
    """
    momentum generation network
    """
    def __init__(self, low_res_factor,opt):
        super(MomentumNet,self).__init__()
        self.low_res_factor = low_res_factor
        """ the low_res_factor control the momentum sz, which should be consistent with the map sz in mermaid unit"""
        using_complex_net = opt[('using_complex_net',True,"using complex version of momentum generation network")]

        if using_complex_net:
            self.mom_gen = MomentumGen_resid(low_res_factor,bn=False)
            print("=================    resid version momentum network is used==============")
        else:
            self.mom_gen = MomentumGen_im(low_res_factor, bn=False)
            print("=================    im version momentum network is used==============")

    def forward(self,input):
        """
        :param input: concatenate of moving and target image
        :return: momentum
        """
        return self.mom_gen(input)


