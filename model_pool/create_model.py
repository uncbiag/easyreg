def create_model(opt):
    model = None
    model_name = opt['model']
    print(model_name)
    if model_name == 'context_net':
        from .context_net import ContextNet
        model = ContextNet()
    elif model_name == 'unet':
        from .unet import Unet
        model = Unet()
    elif opt.model == 'test':
        # from .test_model import TestModel
        # model = TestModel()
        pass
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
