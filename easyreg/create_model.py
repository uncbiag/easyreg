import torch
def create_model(opt):
    """
    create registration model object
    :param opt: ParameterDict, task setting
    :return: model object
    """
    model = None
    model_name = opt['tsk_set']['model']
    gpu_id = opt['tsk_set']['gpu_ids']
    sz = opt
    # gpu_count = torch.cuda.device_count()
    # print("Let's use", min(torch.cuda.device_count(), len(gpu_id) if gpu_id[0]!=-1 else 100), "GPUs!")
    # if gpu_count > 0 and (len(gpu_id) > 1 or gpu_id[0] == -1):
    #     if len(gpu_id) > 1 and gpu_id[0] != -1:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)[1:-1]
    # else:
    #     torch.cuda.set_device(gpu_id[0])
    if gpu_id>=0:
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
    elif model_name == 'seg_net':
        from .seg_net import SegNet
        model = SegNet()
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
