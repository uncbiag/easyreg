from time import time
from pipLine.utils import *




def train_model(opt,model, dataloaders,writer):
    since = time()
    experiment_name = opt['tsk_set']['task_name']
    period = opt['tsk_set'][('print_step', [10,2,1], 'num of steps to print')]
    num_epochs = opt['tsk_set'][('epoch', 100, 'num of epoch')]
    resume_training = opt['tsk_set'][('continue_train', False, 'continue to train')]
    model_path = opt['tsk_set']['path']['model_load_path']
    record_path = opt['tsk_set']['path']['record_path']
    check_point_path = opt['tsk_set']['path']['check_point_path']
    max_batch_num_per_epoch_list = opt['tsk_set']['max_batch_num_per_epoch']
    model.network = model.network.cuda()
    best_model_wts = model.network.state_dict()
    best_model_optimizer = model.optimizer.state_dict() if model.optimizer is not None else None
    best_loss = 0
    epoch_val_loss=0.
    best_epoch = 0
    is_best = False
    start_epoch = 0
    phases =['train','val','debug']
    global_step = {x:0 for x in phases}
    period_loss = {x: 0. for x in phases}
    max_batch_num_per_epoch ={x: max_batch_num_per_epoch_list[i] for i, x in enumerate(phases)}
    period ={x: period[i] for i, x in enumerate(phases)}
    check_best_model_period =opt['tsk_set']['check_best_model_period']
    save_fig_epoch = opt['tsk_set'][('save_val_fig_epoch',10,'saving every num epoch')]
    val_period = opt['tsk_set'][('val_period',10,'saving every num epoch')]
    save_fig_on = opt['tsk_set'][('save_fig_on',True,'saving fig')]



    if resume_training:
        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        start_epoch, best_prec1, global_step=resume_train(model_path, model.network, model.optimizer,old_gpu=old_gpu_id,cur_gpu=cur_gpu_id)
    # else:
    #     model.apply(weights_init)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.set_cur_epoch(epoch)

        # Each epoch has a training and validation phase
        for phase in phases:
            if  phase!='train' and epoch%val_period !=0:
                break
            if not max_batch_num_per_epoch[phase]:
                break
            if phase == 'train':
                model.network.train(True)  # Set model to training mode
            else:
                model.network.train(False)  # Set model to evaluate mode

            running_val_loss =0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                global_step[phase] += 1
                end_of_epoch =   global_step[phase] % max_batch_num_per_epoch[phase] == 0
                is_train = True if phase == 'train' else False
                model.set_input(data,is_train)

                if phase == 'train':
                    model.optimize_parameters()
                    loss = model.get_current_errors()



                elif phase =='val':
                    print('val loss:')
                    model.cal_val_errors()
                    if  epoch>0 and epoch % save_fig_epoch ==0 and save_fig_on:
                        model.save_fig(phase,standard_record=True)
                    loss= model.get_val_res()
                    model.update_loss(epoch,end_of_epoch)
                    running_val_loss += loss



                elif phase == 'debug':
                    print('debugging loss:')
                    model.cal_val_errors()
                    if epoch>0 and epoch % save_fig_epoch ==0 and save_fig_on:
                        model.save_fig(phase,standard_record=True)
                    loss = model.get_val_res()


                period_loss[phase] += loss
                # save for tensorboard, both train and val will be saved

                if global_step[phase] > 1 and global_step[phase] % period[phase] == 0:
                    period_avg_loss = period_loss[phase] / period[phase]
                    writer.add_scalar('loss/' + phase, period_avg_loss, global_step[phase])
                    print("global_step:{}, {} lossing is{}".format(global_step[phase], phase, loss))
                    period_loss[phase] = 0.

                if end_of_epoch:
                    break



            if phase == 'val':
                epoch_val_loss = running_val_loss / min(max_batch_num_per_epoch['val'], dataloaders['data_size']['val'])
                print('{} epoch_val_loss: {:.4f}'.format(epoch, epoch_val_loss))
                if model.exp_lr_scheduler is not None:
                    model.exp_lr_scheduler.step(epoch_val_loss)
                if epoch == 0:
                    best_loss = epoch_val_loss

                if  epoch_val_loss > best_loss:
                    is_best = True
                    best_loss = epoch_val_loss
                    best_epoch = epoch
                    best_model_wts = model.network.state_dict()
                    best_model_optimizer =  model.optimizer.state_dict() if model.optimizer is not None else None


                if epoch % check_best_model_period==0:  #is_best and epoch % check_best_model_period==0:
                    save_checkpoint({'epoch': epoch,'state_dict':  model.network.state_dict(),'optimizer': model.optimizer.state_dict(),
                             'best_loss': best_loss, 'global_step':global_step}, is_best, check_point_path, 'epoch_'+str(epoch), '')
                    is_best = False
                # if epoch % save_visualization_period ==0:
                #     image_summary = model.get_image_summary()
                #     writer.add_image("validation_visualization", image_summary, global_step=global_step)


        print()

    save_checkpoint({'epoch': num_epochs, 'state_dict': model.network.state_dict(),'optimizer': model.optimizer.state_dict(),
                     'best_loss': epoch_val_loss,'global_step':global_step}, False, check_point_path, 'epoch_last', '')

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    writer.close()




    # model.network.load_state_dict(best_model_wts)
    # model.optimizer.load_state_dict(best_model_optimizer)
    running_val_loss = 0.0
    since = time()
    # Iterate over data.
    for data in dataloaders['val']:
        # get the inputs
        is_train =  False
        model.network.train(False)
        model.set_input(data, is_train)
        model.cal_val_errors()
        if save_fig_on:
            model.save_fig('best',standard_record='True')
        loss = model.get_val_res()
        running_val_loss += loss
    val_loss = running_val_loss /  dataloaders['data_size']['val']
    print('the best epoch is {}, the average_val_loss: {:.4f}'.format(best_epoch, val_loss))
    time_elapsed = time() - since
    print('the size of val is {}, val evaluation complete in {:.0f}m {:.0f}s'.format( dataloaders['data_size']['val'],
        time_elapsed // 60, time_elapsed % 60))



    # load best model weights

    return model
