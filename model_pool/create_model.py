import torch
def create_model(opt):
    model = None
    model_name = opt['tsk_set']['model']
    gpu_id = opt['tsk_set']['gpu_ids']
    sz = opt
    torch.cuda.set_device(gpu_id)
    print(model_name)

    ################ models for registration ########################
    if model_name == 'reg_net':
        from .reg_net import RegNet
        model = RegNet()
    elif model_name == 'mermaid_iter':
        from .mermaid_iter import MermaidIter
        model = MermaidIter()
    elif model_name == 'nifty_reg':
        from .nifty_reg_iter import NiftyRegIter
        model = NiftyRegIter()
    elif model_name == 'ants':
        from .ants_iter import AntsRegIter
        model = AntsRegIter()
    elif model_name == 'demons':
        from .demons_iter import DemonsRegIter
        model = DemonsRegIter()
    elif opt.model == 'test':
        # from .test_model import TestModel
        # model = TestModel()
        pass
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
