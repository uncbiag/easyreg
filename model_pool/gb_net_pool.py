import torch
import torch.nn as nn
from torch.autograd import Variable

# 3D UNet and its variants

def encoder(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=True, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.PReLU())
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU())
    return layer


def decoder(in_channels, out_channels, kernel_size, stride=1, padding=0,
            output_padding=0, bias=True, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.PReLU())
    else:
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
    return layer





class UNet_light1(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = encoder(self.in_channel, 16, bias=bias, batchnorm=BN)
        self.ec1 = encoder(16, 32, bias=bias, batchnorm=BN)
        self.ec2 = encoder(32, 32, bias=bias, batchnorm=BN)
        self.ec3 = encoder(32, 64, bias=bias, batchnorm=BN)
        self.ec4 = encoder(64, 64, bias=bias, batchnorm=BN)
        self.ec5 = encoder(64, 128, bias=bias, batchnorm=BN)

        self.pool0 = encoder(32, 32, kernel_size=3, stride=2, padding=1,
            bias=True, batchnorm=True)
        self.pool1 = encoder(64, 64, kernel_size=3, stride=2, padding=1,
            bias=True, batchnorm=True)

        self.dc6 = decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = decoder(64, 64, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # self.weights_init()


    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        e5 = self.ec5(e4)
        del e3, e4

        d6 = torch.cat((self.dc6(e5), syn1), dim=1)
        del e5, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0


class UNet_light2(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = encoder(self.in_channel, 8, bias=bias, batchnorm=BN)
        self.ec1 = encoder(8, 16, bias=bias, batchnorm=BN)
        self.ec2 = encoder(16, 16, bias=bias, batchnorm=BN)
        self.ec3 = encoder(16, 32, bias=bias, batchnorm=BN)
        self.ec4 = encoder(32, 32, bias=bias, batchnorm=BN)
        self.ec5 = encoder(32, 64, bias=bias, batchnorm=BN)

        self.pool0 = encoder(16, 16, kernel_size=3, stride=2, padding=1,
                             bias=True, batchnorm=True)
        self.pool1 = encoder(32, 32, kernel_size=3, stride=2, padding=1,
                             bias=True, batchnorm=True)

        self.dc6 = decoder(64, 64, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = decoder(32, 32, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = decoder(16 + 32, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = decoder(16, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(16, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # self.weights_init()

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        e5 = self.ec5(e4)
        del e3, e4

        d6 = torch.cat((self.dc6(e5), syn1), dim=1)
        del e5, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0




class gbNet(nn.Module):
    """
    A cascaded model from a give model list
    Only train the last model and all other model are pre-trained.
    """
    def __init__(self, model_list,num_class, end2end=False, auto_context=True, residual=False, adaboost=False, residual_scale=1.0):
        super(gbNet, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.num_models = len(self.models)
        self.end2end = end2end
        self.auto_context = auto_context
        self.residual = residual
        self.residual_scale = residual_scale
        self.adaboost = adaboost
        assert (not residual&adaboost) & (residual or adaboost), "one mode and only one mode should be on"
        self.cur_model_id = -1
        self.num_class =num_class
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.st_copy_previous_model = 0


    def weights_init(self):
        if self.end2end:
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    if not m.weight is None:
                        nn.init.xavier_normal(m.weight.data)
                    if not m.bias is None:
                        m.bias.data.zero_()
        else:
            for m in self.models[-1].modules():
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    if not m.weight is None:
                        nn.init.xavier_normal(m.weight.data)
                    if not m.bias is None:
                        m.bias.data.zero_()

    def cascaded_parameters(self):
        if self.end2end:
            return self.parameters()
        else:
            return self.models[-1].parameters()



    def _set_cur_model_id(self,model_id):
        """
        :param model_id:  the mode id should be 0 .... num_models-1
        :return:
        """
        self.cur_model_id = model_id
        self.num_cur_models = self.cur_model_id+1



    def _init_cur_training_model(self):
        """
        Assume all the net[0,..model_id-1] has already been trained
        :return:
        """
        model_id = self.cur_model_id
        if not self.end2end:
            # the model before the current model will be in the inference state
            if model_id>0:
                # for param in self.models[self.cur_model_id-1].parameters():
                #     param.requires_grad = False
                self.models[self.cur_model_id-1].eval()
            # we use the previous model's parameter to initialized the current model to be trained
                if model_id > self.st_copy_previous_model:
                    self.models[model_id].load_state_dict(self.models[model_id-1].state_dict())
            self.models[model_id].train(True)



    def set_cascaded_eval(self):
        # this function will not be used
        self.training = False
        self._set_cur_model_id(self.num_models-1)
        self.eval()


    def set_cascaded_train(self, model_id):
        """
        :param model_id: the model id should be 0,1,2,3,...

        :return:
        """
        self.training = True
        self._set_cur_model_id(model_id)
        self._init_cur_training_model()


    def encode_target(self, in_sz, target):
        from functools import reduce
        extra_dim = reduce(lambda x, y: x * y, in_sz[2:])
        targ_one_hot = Variable(torch.ones(in_sz[0], in_sz[1], extra_dim)).cuda() * (-1 / (self.num_class - 1))
        targ_one_hot.scatter_(1, target.view(in_sz[0], 1, extra_dim), 1.)
        target = targ_one_hot.view(in_sz).contiguous()
        return target


    def adaboost_module(self, logit, target=None):
        learning_rate = 1
        logp_output = self.log_softmax(logit)
        weight = None
        h = None
        if self.training:
            weight = torch.exp(-learning_rate * (((self.num_class - 1.) / self.num_class)
                                * torch.sum(target * logp_output, dim=1)))
        else:
            h = (self.num_class - 1) * (logp_output - (1. / self.num_class)
                                * torch.sum(logp_output, dim=1,keepdim=True))
        return weight, h

    def check_if_volatile(self, id, num_models):
        volatile = True if id < num_models - 1 and not self.end2end else not self.training
        return volatile


    def forward(self, input, target=None):
        """
        Forward though the cascased models.
        If using residual, each sub-model's output is added to the output of previous models
        If using auto-context, each sub-model's input is the concatenation of the raw input
        and the output of the previous sub-model
        :param input: input for the first model
        :param train: if training mode
        :return:the output of the last model
        """


        train = self.training
        num_cur_models = self.num_cur_models
        estimator_weight = None
        h = None
        resid_output = None
        if self.cur_model_id ==0:
            input.volatile= False
        output = self.models[0](input)
        in_sz = list(output.size())

        if self.adaboost:
            if train:
                target = self.encode_target(in_sz, target)
            estimator_weight, h = self.adaboost_module(output, target=target)


        if self.residual:
            resid_output = output


        for i in range(1, num_cur_models):
            volatile = self.check_if_volatile(i, num_cur_models)
            if self.auto_context:
                # voliate on  when is (not the current model, not end to end) or not train
                temp_input = Variable(torch.cat([self.softmax(output).data, input.data], dim=1),
                                      volatile=volatile)
            else:
                temp_input = Variable(input.data.cuda(), volatile=volatile)


            # detach the output of the last model
            if i == num_cur_models-1 and not self.end2end and train:
                output = output.detach()
                output.volatile=False

            output = self.models[i](temp_input)

            if self.residual:
                resid_output = resid_output*self.residual_scale + output


            if self.adaboost:
                if train:
                    # the weight of the current model is not used
                    if i < num_cur_models-1:
                        cur_w, cur_h = self.adaboost_module(output, target=target)
                        estimator_weight *= cur_w
                else:
                    cur_w, cur_h = self.adaboost_module(output, target=target)
                    h += cur_h

        if train:
            if self.residual:
                return resid_output, None

            elif self.adaboost:
                if self.cur_model_id>0:
                    estimator_weight.volatile = False

                return output, estimator_weight
        else:
            if self.residual:
                return resid_output
            elif self.adaboost:
                return h




