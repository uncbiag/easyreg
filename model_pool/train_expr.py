from time import time
from model_pool.net_utils import *




def train_model(opt,model, dataloaders,writer):
    since = time()
    print_step = opt['tsk_set'][('print_step', [10,4,4], 'num of steps to print')]
    num_epochs = opt['tsk_set'][('epoch', 100, 'num of training epoch')]
    continue_train = opt['tsk_set'][('continue_train', False, 'continue to train')]
    model_path = opt['tsk_set']['path']['model_load_path']
    load_model_but_train_from_begin = opt['tsk_set'][('load_model_but_train_from_begin',False,'load_model_but_train_from_begin')]
    load_model_but_train_from_epoch =opt['tsk_set'][('load_model_but_train_from_epoch',0,'load_model_but_train_from_epoch')]
    check_point_path = opt['tsk_set']['path']['check_point_path']
    max_batch_num_per_epoch_list = opt['tsk_set'][('max_batch_num_per_epoch',(-1,-1,-1,-1),"max batch number per epoch for train|val|test|debug")]
    gpu_id = opt['tsk_set']['gpu_ids']
    best_score = 0
    is_best = False
    start_epoch = 0
    best_epoch = -1
    phases =['train','val','debug']
    global_step = {x:0 for x in phases}
    period_loss = {x: 0. for x in phases}
    period_detailed_scores = {x: 0. for x in phases}
    max_batch_num_per_epoch ={x: max_batch_num_per_epoch_list[i] for i, x in enumerate(phases)}
    period ={x: print_step[i] for i, x in enumerate(phases)}
    check_best_model_period =opt['tsk_set'][('check_best_model_period',5,'save best performed model every # epoch')]
    tensorboard_print_period = { phase: min(max_batch_num_per_epoch[phase],period[phase]) for phase in phases}
    save_fig_epoch = opt['tsk_set'][('save_val_fig_epoch',2,'saving every num epoch')]
    save_3d_img_on = opt['tsk_set'][('save_3d_img_on', False, 'saving fig')]
    val_period = opt['tsk_set'][('val_period',10,'do validation every num epoch')]
    save_fig_on = opt['tsk_set'][('save_fig_on',True,'saving fig')]
    warmming_up_epoch = opt['tsk_set'][('warmming_up_epoch',2,'warming up the model in the first # epoch')]
    continue_train_lr = opt['tsk_set'][('continue_train_lr', -1, 'learning rate for continuing to train')]
    opt['tsk_set']['optim']['lr'] =opt ['tsk_set']['optim']['lr'] if not continue_train else continue_train_lr


    if continue_train:
        start_epoch, best_prec1, global_step=resume_train(model_path, model.network,model.optimizer)
        if continue_train_lr > 0:
            model.update_learning_rate(continue_train_lr)
            print("the learning rate has been changed into {} when resuming the training".format(continue_train_lr))
        if load_model_but_train_from_begin:
            start_epoch=load_model_but_train_from_epoch
            global_step = {x: load_model_but_train_from_epoch*max_batch_num_per_epoch[x] for x in phases}
            print("the model has been initialized from extern, but will train from the epoch {}".format(start_epoch))
    #
    # gpu_count = torch.cuda.device_count()
    #
    # if gpu_count>0 and (len( gpu_id)>1 or gpu_id[0]==-1):
    #     model.network = nn.DataParallel(model.network)
    #     model.set_multi_gpu_on()
    #     #model.network = model.network.module
    #     model.network.cuda()
    # else:
    #     model.network = model.network.cuda()
    if gpu_id>=0:
        model.network = model.network.cuda()

    for epoch in range(start_epoch, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.set_cur_epoch(epoch)
        if epoch == warmming_up_epoch:
            model.update_learning_rate()

        for phase in phases:
            # if is not training phase, and not the #*val_period , then break
            if  phase!='train' and epoch%val_period !=0:
                break
            # if # = 0 or None then skip the val or debug phase
            if not max_batch_num_per_epoch[phase]:
                break
            if phase == 'train':
                model.update_scheduler(epoch)
                model.set_train()
            elif phase == 'val':
                model.set_val()
            else:
                model.set_debug()

            running_val_score =0.0
            running_debug_score =0.0

            for data in dataloaders[phase]:

                global_step[phase] += 1
                end_of_epoch = global_step[phase] % max_batch_num_per_epoch[phase] == 0
                is_train = True if phase == 'train' else False
                model.set_input(data,is_train)
                loss = 0.
                detailed_scores = 0.

                if phase == 'train':
                    # from mermaid.utils import time_warped_function
                    # optimize_parameters = time_warped_function(model.optimize_parameters)
                    # optimize_parameters()
                    model.optimize_parameters()

                    # try:
                    #     model.optimize_parameters()
                    # except:
                    #     info = model.get_debug_info()
                    #     save_and_debug_model(model,info,check_point_path,epoch,global_step)
                    #     exit(1)
                    loss = model.get_current_errors()



                elif phase =='val':
                    model.cal_val_errors()
                    if epoch % save_fig_epoch ==0 and save_fig_on:
                        model.save_fig(phase)
                        if save_3d_img_on:
                            model.save_fig_3D(phase='val')
                    score, detailed_scores= model.get_val_res()
                    print('val loss of batch {} is {}:'.format(model.get_image_names(),score))
                    model.update_loss(epoch,end_of_epoch)
                    running_val_score += score
                    loss = score



                elif phase == 'debug':
                    print('debugging loss:')
                    model.cal_val_errors()
                    if epoch>0 and epoch % save_fig_epoch ==0 and save_fig_on:
                        model.save_fig(phase)
                        if save_3d_img_on:
                            model.save_fig_3D(phase='debug')
                    score, detailed_scores = model.get_val_res()
                    running_debug_score += score
                    loss = score

                model.do_some_clean()

                # save for tensorboard, both train and val will be saved
                period_loss[phase] += loss
                if not is_train:
                    period_detailed_scores[phase] += detailed_scores

                if global_step[phase] > 0 and global_step[phase] % tensorboard_print_period[phase] == 0:
                    if not is_train:
                        period_avg_detailed_scores = np.squeeze(period_detailed_scores[phase]) / tensorboard_print_period[phase]
                        for i in range(len(period_avg_detailed_scores)):
                            writer.add_scalar('loss/'+ phase+'_l_{}'.format(i), period_avg_detailed_scores[i], global_step['train'])
                        period_detailed_scores[phase] = 0.

                    period_avg_loss = period_loss[phase] / tensorboard_print_period[phase]
                    writer.add_scalar('loss/' + phase, period_avg_loss, global_step['train'])
                    print("global_step:{}, {} lossing is{}".format(global_step['train'], phase, period_avg_loss))
                    period_loss[phase] = 0.

                if end_of_epoch:
                    break

            if phase == 'val':
                epoch_val_score = running_val_score / min(max_batch_num_per_epoch['val'], dataloaders['data_size']['val'])
                print('{} epoch_val_score: {:.4f}'.format(epoch, epoch_val_score))
                if model.exp_lr_scheduler is not None:
                    model.exp_lr_scheduler.step(epoch_val_score)
                    print("debugging, the exp_lr_schedule works and update the step")
                if epoch == 0:
                    best_score = epoch_val_score

                if epoch_val_score > best_score:
                    is_best = True
                    best_score = epoch_val_score
                    best_epoch = epoch
            if phase == 'train':
                # currently we just save model by period, so need to check the best model manually
                if epoch % check_best_model_period==0:  #is_best and epoch % check_best_model_period==0:
                    if isinstance(model.optimizer, tuple):
                        # for multi-optimizer cases,
                        optimizer_state = []
                        for term in model.optimizer:
                            optimizer_state.append(term.state_dict())
                        optimizer_state = tuple(optimizer_state)
                    else:
                        optimizer_state  = model.optimizer.state_dict()
                    save_checkpoint({'epoch': epoch,'state_dict':  model.network.state_dict(),'optimizer': optimizer_state,
                             'best_score': best_score, 'global_step':global_step}, is_best, check_point_path, 'epoch_'+str(epoch), '')
                    is_best = False


            if phase == 'debug':
                epoch_debug_score = running_debug_score / min(max_batch_num_per_epoch['debug'], dataloaders['data_size']['debug'])
                print('{} epoch_debug_score: {:.4f}'.format(epoch, epoch_debug_score))




    # optional,  at end of the training, lets save the last model and optimizer
    # if isinstance(model.optimizer, tuple):
    #     optimizer_state = []
    #     for term in model.optimizer:
    #         optimizer_state.append(term.state_dict())
    #     optimizer_state = tuple(optimizer_state)
    # else:
    #     optimizer_state = model.optimizer.state_dict()
    # save_checkpoint({'epoch': num_epochs, 'state_dict': model.network.state_dict(), 'optimizer': optimizer_state,
    #                  'best_score': epoch_val_score, 'global_step': global_step}, False, check_point_path, 'epoch_'+str(num_epochs),
    #                 '')

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val score : {:4f} is at epoch {}'.format(best_score, best_epoch))
    writer.close()
    # return the model at the last epoch, not the best epoch
    return model


def save_and_debug_model(model, info,check_point_path,epoch,global_step):
    print("the program meet error, now output the debugging info")
    print("{}".format(info))
    if isinstance(model.optimizer, tuple):
        # for multi-optimizer cases
        optimizer_state = []
        for term in model.optimizer:
            optimizer_state.append(term.state_dict())
        optimizer_state = tuple(optimizer_state)
    else:
        optimizer_state = model.optimizer.state_dict()
    save_checkpoint({'epoch': epoch, 'state_dict': model.network.state_dict(), 'optimizer': optimizer_state,
                     'best_score': 0.0, 'global_step': global_step}, False, check_point_path,
                    'epoch_' + 'debug', '')
