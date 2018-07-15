from time import time, sleep
from pipLine.utils import *




def train_model(opt,model, dataloaders,writer):
    since = time()
    experiment_name = opt['tsk_set']['task_name']
    dataset_name = opt['tsk_set']['dataset_name']

    print_step = opt['tsk_set'][('print_step', [10,4,4], 'num of steps to print')]
    num_epochs = opt['tsk_set'][('epoch', 100, 'num of epoch')]
    resume_training = opt['tsk_set'][('continue_train', False, 'continue to train')]
    model_path = opt['tsk_set']['path']['model_load_path']
    record_path = opt['tsk_set']['path']['record_path']
    check_point_path = opt['tsk_set']['path']['check_point_path']
    max_batch_num_per_epoch_list = opt['tsk_set']['max_batch_num_per_epoch']
    #model.network = model.network.cuda()
    #best_model_wts = model.network.state_dict()
    #best_model_optimizer = model.optimizer.state_dict() if model.optimizer is not None else None
    best_loss = 0
    epoch_val_loss=0.
    epoch_debug_loss =0.
    best_epoch = 0
    is_best = False
    start_epoch = 0
    phases =['train','val','debug']
    global_step = {x:0 for x in phases}
    period_loss = {x: 0. for x in phases}
    period_full_loss = {x: 0. for x in phases}
    max_batch_num_per_epoch ={x: max_batch_num_per_epoch_list[i] for i, x in enumerate(phases)}
    period ={x: print_step[i] for i, x in enumerate(phases)}
    check_best_model_period =opt['tsk_set']['check_best_model_period']
    effective_print_period = { phase: min(max_batch_num_per_epoch[phase],period[phase]) for phase in phases}
    save_fig_epoch = opt['tsk_set'][('save_val_fig_epoch',2,'saving every num epoch')]
    val_period = opt['tsk_set'][('val_period',10,'saving every num epoch')]
    save_fig_on = opt['tsk_set'][('save_fig_on',True,'saving fig')]
    warmming_up_epoch = opt['tsk_set'][('warmming_up_epoch',2,'warmming_up_epoch')]
    continue_train_lr = opt['tsk_set'][('continue_train_lr', -1, 'continue to train')]
    update_model_by_val = opt['tsk_set'][('update_model_by_val', True, 'update_model_by_val')]
    tor_thre = opt['tsk_set'][('tor_thre', 0.8, 'tor_thre')]
    update_model_torl = opt['tsk_set'][('update_model_torl', 2, 'update_model_torl')]
    update_model_epoch_torl = opt['tsk_set'][('update_model_epoch_torl', 80, 'update_model_epoch_torl')]
    update_model_epoch_count = 0
    epoch_val_record = -1
    epoch_debug_record = -1
    model_torl_count = 0
    update_model_epoch_best_record= 0
    update_model_epoch_best_dice_record=0.
    last_met = True
    stop_train = False




    if resume_training:
        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        start_epoch, best_prec1, global_step=resume_train(model_path, model.network,None,old_gpu=old_gpu_id,cur_gpu=cur_gpu_id)
        if continue_train_lr>0:
            model.adjust_learning_rate(continue_train_lr)
            print("the learning rate has been changed into {} when resuming the training".format(continue_train_lr))

    # else:
    #     model.apply(weights_init)

    model.network = model.network.cuda()

    for epoch in range(start_epoch, num_epochs+1):
        update_model_epoch_count += 1
        if stop_train:
            print("the model meets validation requirements and finished training ")
            break

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.set_cur_epoch(epoch)
        if epoch == warmming_up_epoch:
            model.adjust_learning_rate()

        for phase in phases:
            # if is not training phase, and not the #*val_period , then break
            if  phase!='train' and epoch%val_period !=0:
                break
            # if # = 0, then skip the val or debug phase
            if not max_batch_num_per_epoch[phase]:
                break
            if phase == 'train':
                #model.network.train(True)  # Set model to training mode
                model.set_train()
            else:
                #model.network.train(False)  # Set model to evaluate mode
                if phase =='val':
                    model.set_val()
                else:
                    model.set_debug()

            running_val_loss =0.0
            running_debug_loss =0.0

            # Iterate over data.
            time_cc=0
            for data in dataloaders[phase]:
                # get the inputs
                # if time_cc%10==0:
                #     print("this is just for overloading speed,time_cc{}".format(time_cc))
                #     print(data[0]['image'].shape)
                # time_cc +=1
                # sleep(0.2)
                # continue

                global_step[phase] += 1
                end_of_epoch = global_step[phase] % max_batch_num_per_epoch[phase] == 0
                is_train = True if phase == 'train' else False
                model.set_input(data,is_train)

                if phase == 'train':
                    model.optimize_parameters()
                    loss = model.get_current_errors()



                elif phase =='val':
                    print('val loss:')
                    model.cal_val_errors()
                    if  epoch>0 and epoch % save_fig_epoch ==0 and save_fig_on:
                        model.save_fig(phase,standard_record=False)
                    loss, full_loss= model.get_val_res()
                    model.update_loss(epoch,end_of_epoch)
                    running_val_loss += loss



                elif phase == 'debug':
                    print('debugging loss:')
                    model.cal_val_errors()
                    if epoch>0 and epoch % save_fig_epoch ==0 and save_fig_on:
                        model.save_fig(phase,standard_record=False)
                    loss, full_loss = model.get_val_res()
                    running_debug_loss += loss

                model.do_some_clean()

                # save for tensorboard, both train and val will be saved
                period_loss[phase] += loss
                if not is_train:
                    period_full_loss[phase] += full_loss

                if global_step[phase] > 0 and global_step[phase] % effective_print_period[phase] == 0:
                    if not is_train:
                        period_avg_full_loss = np.squeeze(period_full_loss[phase]) / effective_print_period[phase]
                        for i in range(len(period_avg_full_loss)):
                            writer.add_scalar('loss/'+ phase+'_l_{}'.format(i), period_avg_full_loss[i], global_step['train'])
                        period_full_loss[phase] = 0.

                    period_avg_loss = period_loss[phase] / effective_print_period[phase]
                    writer.add_scalar('loss/' + phase, period_avg_loss, global_step['train'])
                    print("global_step:{}, {} lossing is{}".format(global_step['train'], phase, period_avg_loss))
                    period_loss[phase] = 0.

                if end_of_epoch:
                    break



            if phase == 'val':
                epoch_val_loss = running_val_loss / min(max_batch_num_per_epoch['val'], dataloaders['data_size']['val'])
                print('{} epoch_val_loss: {:.4f}'.format(epoch, epoch_val_loss))
                if model.exp_lr_scheduler is not None:
                    model.exp_lr_scheduler.step(epoch_val_loss)
                    print("debugging, the exp_lr_schedule works and update the step")
                if epoch == 0:
                    best_loss = epoch_val_loss

                if epoch_val_loss > best_loss:
                    is_best = True
                    best_loss = epoch_val_loss
                    best_epoch = epoch
                if update_model_by_val and epoch>1:
                    better_res= epoch_val_loss > update_model_epoch_best_dice_record
                    update_model_epoch_best_dice_record = epoch_val_loss if better_res else update_model_epoch_best_dice_record
                    update_model_epoch_best_record = epoch if better_res else update_model_epoch_best_record
                    if np.abs(epoch_val_loss - epoch_val_record)*100 < tor_thre or update_model_epoch_count> update_model_epoch_torl:
                        model_torl_count *= int(last_met)
                        model_torl_count += 1
                        if model_torl_count >= update_model_torl or  update_model_epoch_count> update_model_epoch_torl:
                            if model_torl_count >= update_model_torl:
                                print("reached the update_model_torl :{}".format(update_model_torl))
                            else:
                                print("reached the update epoch_torl :{}".format(update_model_epoch_torl))
                            stop_train =model.check_and_update_model(update_model_epoch_best_record)
                            model_torl_count =0
                            update_model_epoch_count =0
                            update_model_epoch_best_dice_record=0
                    last_met = np.abs(epoch_val_loss - epoch_val_record)*100 < tor_thre
                    epoch_val_record = epoch_val_loss

            if phase == 'train':
                if epoch % check_best_model_period==0:  #is_best and epoch % check_best_model_period==0:
                    if isinstance(model.optimizer, tuple):
                        optimizer_state = []
                        for term in model.optimizer:
                            optimizer_state.append(term.state_dict())
                        optimizer_state = tuple(optimizer_state)
                    else:
                        optimizer_state  = model.optimizer.state_dict()
                    save_checkpoint({'epoch': epoch,'state_dict':  model.network.state_dict(),'optimizer': optimizer_state,
                             'best_loss': best_loss, 'global_step':global_step}, is_best, check_point_path, 'epoch_'+str(epoch), '')
                    is_best = False


            if phase == 'debug':

                epoch_debug_loss = running_debug_loss / min(max_batch_num_per_epoch['debug'], dataloaders['data_size']['debug'])
                print('{} epoch_debug_loss: {:.4f}'.format(epoch, epoch_debug_loss))
                if not update_model_by_val and epoch >1:
                    better_res= epoch_val_loss > update_model_epoch_best_dice_record
                    update_model_epoch_best_dice_record = epoch_val_loss if better_res  else update_model_epoch_best_dice_record
                    update_model_epoch_best_record = epoch if better_res else update_model_epoch_best_record
                    if np.abs(epoch_debug_loss - epoch_debug_record) * 100 < tor_thre or update_model_epoch_count> update_model_epoch_torl:
                        model_torl_count *= int(last_met)
                        model_torl_count += 1
                        if model_torl_count >= update_model_torl or update_model_epoch_count> update_model_epoch_torl: #and  epoch_debug_loss > epoch_debug_record:
                            if model_torl_count >= update_model_torl:
                                print("reached the update_model_torl :{}".format(update_model_torl))
                            else:
                                print("reached the update epoch_torl :{}".format(update_model_epoch_torl))
                            stop_train = model.check_and_update_model(update_model_epoch_best_record)
                            model_torl_count = 0
                            update_model_epoch_count = 0
                            update_model_epoch_best_dice_record=0
                    last_met = np.abs(epoch_debug_loss - epoch_debug_record) * 100 < tor_thre
                    epoch_debug_record = epoch_debug_loss

        print()

    if isinstance(model.optimizer, tuple):
        optimizer_state = []
        for term in model.optimizer:
            optimizer_state.append(term.state_dict())
        optimizer_state = tuple(optimizer_state)
    else:
        optimizer_state = model.optimizer.state_dict()
    save_checkpoint({'epoch': num_epochs, 'state_dict': model.network.state_dict(), 'optimizer': optimizer_state,
                     'best_loss': epoch_val_loss, 'global_step': global_step}, False, check_point_path, 'epoch_'+str(num_epochs),
                    '')
    save_checkpoint({'epoch': num_epochs, 'state_dict': model.network.state_dict(),'optimizer':optimizer_state,
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
    model.set_val()
    # Iterate over data.
    for data in dataloaders['val']:
        # get the inputs
        is_train =  False
        model.network.train(False)
        model.set_input(data, is_train)
        model.cal_val_errors()
        if save_fig_on:
            model.save_fig('best',standard_record='True')
        loss, full_loss = model.get_val_res()
        running_val_loss += loss
    val_loss = running_val_loss /  dataloaders['data_size']['val']
    print('the best epoch is {}, the average_val_loss: {:.4f}'.format(best_epoch, val_loss))
    time_elapsed = time() - since
    print('the size of val is {}, val evaluation complete in {:.0f}m {:.0f}s'.format( dataloaders['data_size']['val'],
        time_elapsed // 60, time_elapsed % 60))


    running_val_loss = 0.0
    # do the test data evaluation
    if not dataset_name =='brats':
        for data in dataloaders['test']:
            is_train =  False
            model.network.train(False)
            model.set_input(data, is_train)
            model.cal_val_errors()
            if save_fig_on:
                model.save_fig('best',standard_record='True')
            loss, full_loss = model.get_val_res()
            running_val_loss += loss
        test_loss = running_val_loss /  dataloaders['data_size']['test']
        print('the best epoch is {}, the average_test_loss: {:.4f}'.format(best_epoch, test_loss))
        time_elapsed = time() - since
        print('the size of test is {}, test evaluation complete in {:.0f}m {:.0f}s'.format( dataloaders['data_size']['test'],
            time_elapsed // 60, time_elapsed % 60))
    else:
        dg_key_word = ''
        for data in dataloaders['test']:
            # get the inputs
            is_train = False
            model.network.train(False)
            model.set_input(data, is_train)
            model.get_pred_img(split_size=3)
            # model.get_output_map()
            if save_fig_on:
                model.save_fig(dg_key_word, standard_record=True, saving_gt=False)




    return model
