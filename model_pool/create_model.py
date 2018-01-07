def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'context_net':
        from .context_net import ContextNet
        model = ContextNet()
    elif opt.model == 'unet':
        from .unet import Unet
        model = Unet()
    elif opt.model == 'test':
        # from .test_model import TestModel
        # model = TestModel()
        pass
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
