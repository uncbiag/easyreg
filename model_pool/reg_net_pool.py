from model_pool.reg_net import RegNet


class SimpleNet(RegNet):
    def initialize(self,opt):
        RegNet().initialize(opt)
