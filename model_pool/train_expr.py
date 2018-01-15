from time import time
from pipLine.utils import *




def train_model(opt,model, dataloaders,writer):
    since = time()
    experiment_name = opt['tsk_set']['task_name']
    period = opt['tsk_set'][('print_step', 10, 'num of steps to print')]
    num_epochs = opt['tsk_set'][('epoch', 100, 'num of epoch')]
    resume_training = opt['tsk_set'][('continue_train', False, 'continue to train')]
    model_path = opt['tsk_set'][('model_path', '', 'if continue_train, given the model path')]
    record_path = opt['tsk_set']['path']['record_path']
    check_point_path = opt['tsk_set']['path']['check_point_path']
    max_batch_num_per_epoch_list = opt['tsk_set']['max_batch_num_per_epoch']
    best_model_wts = model.network.state_dict()
    best_loss = 0
    best_epoch = 0
    is_best = False
    model.network = model.network.cuda()
    start_epoch = 0
    phases =['train','val','debug']
    global_step = {x:0 for x in phases}
    period_loss = {x: 0. for x in phases}
    max_batch_num_per_epoch ={x: max_batch_num_per_epoch_list[i] for i, x in enumerate(phases)}
    check_best_model_period =opt['tsk_set']['check_best_model_period']
    save_visualization_period = opt['tsk_set'][('save_visualization_period',10,'saving every num epoch')]



    if resume_training:
        start_epoch, best_loss =resume_train(model_path, model.network)
    # else:
    #     model.apply(weights_init)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            save_per_epoch = 1
            if phase == 'train':
                model.network.train(True)  # Set model to training mode
            else:
                model.network.train(False)  # Set model to evaluate mode

            running_val_loss =0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                is_train = True if phase == 'train' else False
                model.set_input(data,is_train)

                if phase == 'train':
                    model.optimize_parameters()
                    loss = model.get_current_errors()

                elif phase =='val':
                    print('val loss:')
                    model.cal_val_errors()
                    #model.save_val_fig(phase)
                    loss= model.get_val_res()
                    running_val_loss += loss
                    
                elif phase == 'debug':
                    print('debugging loss:')
                    model.cal_val_errors()
                    #model.save_val_fig(phase)
                    loss = model.get_val_res()


                period_loss[phase] += loss
                # save for tensorboard, both train and val will be saved
                global_step[phase] += 1
                if global_step[phase] > 1 and global_step[phase] % period == 0:
                    period_avg_loss = period_loss[phase] / period
                    writer.add_scalar('loss/' + phase, period_avg_loss, global_step[phase])
                    print("global_step:{}, {} lossing is{}".format(global_step[phase], phase, loss))
                    period_loss[phase] = 0.

                if global_step[phase] % max_batch_num_per_epoch[phase] == 0:
                    break



            if phase == 'val':
                epoch_val_loss = running_val_loss / min(max_batch_num_per_epoch['val'], dataloaders['data_size'][phase])
                print('{} epoch_val_loss: {:.4f}'.format(epoch, epoch_val_loss))

                if epoch == 0:
                    best_loss = epoch_val_loss

                if  epoch_val_loss > best_loss:
                    is_best = True
                    best_loss = epoch_val_loss
                    best_epoch = epoch
                    best_model_wts = model.network.state_dict()

                if is_best and epoch % check_best_model_period==0:
                    save_checkpoint({'epoch': epoch,'state_dict': model.network.state_dict(),
                             'best_loss': best_loss}, is_best, check_point_path, 'epoch_'+str(best_epoch), '')
                    is_best = False
                # if epoch % save_visualization_period ==0:
                #     image_summary = model.get_image_summary()
                #     writer.add_image("validation_visualization", image_summary, global_step=global_step)


        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    writer.close()

    # load best model weights
    #model.network.load_state_dict(best_model_wts)
    return model
