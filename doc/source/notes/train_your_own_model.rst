Train your own model
========================================


Quick Start
____________

In this tutorial, we would show how to train your own model.


About how to prepare the data, please refer to the **Prepare Data** section.


The script *demo_for_easyreg_train.py* is for training new learning-based registration model.

::

        A training interface for learning methods.
        The method support list :  mermaid-related methods
        Assume there is three-level folder, output_root_path/ data_task_folder/ task_folder
        Arguments:
            --output_root_path/ -o: the path of output folder
            --data_task_name/ -dtn: data task name i.e. lung_reg_task , oai_reg_task
            --task_name / -tn: task name i.e. run_training_vsvf_task, run_training_rdmm_task
            --setting_folder_path/ -ts: path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings.json(optional) and mermaid_nonp_settings(optional)
            --train_affine_first: train affine network first, then train non-parametric network
            --gpu_id/ -g: on which gpu to run

An example

..  code::

    python demo_for_easyreg_train.py -o=/playpen/zyshen/data -data_task_name=croped_for_reg_debug_3000_pair_oai_reg_inter -task_name=interface_rdmm -ts=/playpen/zyshen/reg_clean/demo/demo_settings/mermaid/training_network_rdmm -g=0


In the 'task_name' folder, three folder will be auto created, **log** for tensorboard, **checkpoints** for saving models,
**records** for saving running time results. Besides, two files will also be created. **task_settings.json** for recording settings of current tasks.
**logfile.log** for terminal output ( only flushed when task finished).


* One thing to mention is that the affine-network involves the fully connection layer,  whose input channel number needs to be adjusted by the input image size.


Training Settings
__________________

Because this projects involves two repositories, the EasyReg and the Mermaid.
There are two json files needs to be set: ``cur_task_setting.json`` for EasyReg and  ``mermaid_nonp_settings.json`` for Mermaid.
Here we would provide document on both json files and corresponding demos (can be found in **demo** directory).


Settings for EasyReg
^^^^^^^^^^^^^^^^^^^^^

The detailed training settings can be found in ``cur_task_setting_comment.json``, which is shared by all mermaid-based models.


.. code::

    {
        "dataset": {
            "img_after_resize": "image size after resampling",
            "max_pair_for_loading": "the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]",
            "load_training_data_into_memory": "when train network, load all training sample into memory can relieve disk burden",
            "spacing_to_refer": "the physical spacing in numpy coordinate, only activate when using_physical_coord is true"
        },
        "tsk_set": {
            "batch_sz": "batch sz (only for mermaid related method, otherwise set to 1)",
            "check_best_model_period":"save best performed model every # epoch",
            "continue_train": "for network training method, continue training the model loaded from model_path",
            "continue_train_lr": "learning rate for continuing to train",
            "criticUpdates": "for network training method, the num determines gradient update every # iter",
            "epoch": "num of training epoch",
            "gpu_ids": "the gpu id used for network methods",
            "loss": {
                "type": "the similarity measure type"
            },
            "max_batch_num_per_epoch": "max batch number per epoch for train|val|test|debug",
            "model_path": "if continue_train, the model path should be given here",
            "n_in_channel": "for network training method, the color channel typically set to 1",
            "optim": {
                "adam": {},
                "lr_scheduler": {
                    "custom": {}
                }
            },
            "output_orginal_img_type": "output follows the same sz and physical property of the original image (input by command line or txt)",
            "path": {
                "__doc__": "record paths"
            },
            "reg": {
                "affine_net": {
                    "acc_multi_step_loss": "accumulate loss at each step",
                    "affine_net_iter": "num of step",
                    "epoch_activate_extern_loss": "epoch to activate the external loss which will replace the default ncc loss",
                    "epoch_activate_multi_step": "epoch to activate multi-step affine",
                    "epoch_activate_sym": "epoch to activate symmetric forward",
                    "epoch_activate_sym_loss": "the epoch to take symmetric loss into backward , only if epoch_activate_sym and epoch_activate_sym_loss",
                    "lr_for_multi_step": "if reset_lr_for_multi_step, reset learning rate into # when multi-step begins",
                    "initial_reg_factor": "initial regularization factor",
                    "min_reg_factor": "minimum regularization factor",
                    "reset_lr_for_multi_step": "if True, reset learning rate when multi-step begins",
                    "using_complex_net": "use complex version of affine net"
                },
                "compute_inverse_map": "compute the inverse transformation map",
                "low_res_factor": "factor of low-resolution map",
                "mermaid_net": {
                    "affine_init_path": "the path of trained affined network",
                    "affine_refine_step": "the multi-step num in affine refinement",
                    "clamp_momentum": "clamp_momentum",
                    "clamp_thre": "clamp momentum into [-clamp_thre, clamp_thre]",
                    "epoch_activate_multi_step": "epoch activate the multi-step",
                    "epoch_activate_sym": "epoch activate the symmetric loss",
                    "epoch_list_fixed_deep_smoother_network": "epoch_list_fixed_deep_smoother_network",
                    "epoch_list_fixed_momentum_network": "list of epoch, fix the momentum network",
                    "load_trained_affine_net": "if true load_trained_affine_net; if false, the affine network is not initialized",
                    "lr_for_multi_step": "if reset_lr_for_multi_step, reset learning rate when multi-step begins",
                    "mermaid_net_json_pth": "the path for mermaid settings json",
                    "num_step": "compute multi-step loss",
                    "optimize_momentum_network": "if true, optimize the momentum network",
                    "reset_lr_for_multi_step": "if True, reset learning rate when multi-step begins",
                    "sym_factor": "factor on symmetric loss",
                    "using_affine_init": "if ture, deploy an affine network before mermaid-net",
                    "using_physical_coord": "use physical coordinate system",
                    "using_complex_net": "using complex version of momentum generation network"
                }
            },
            "save_3d_img_on": "saving fig",
            "save_extra_3d_img": "save extra image",
            "save_fig_on": "saving fig",
            "train": "if is in train mode",
            "use_physical_coord": "Keep physical spacing",
            "val_period": "do validation every num epoch",
            "warmming_up_epoch": "warming up the model in the first # epoch"
          }
    }



Settings for Mermaid
^^^^^^^^^^^^^^^^^^^^^^
The corresponding comments for Meramid part are in ``mermaid_nonp_settins_comment.json``.
Depends on model and similarity measure, the **comments** may differ.

Here we list setting typical setting documents on vSVF model and RDMM model.

**Mermaid settings on vSVF**

..   code::

    {
        "model": {
            "deformation": {
                "compute_similarity_measure_at_low_res": "to compute Sim at lower resolution"
            },
            "registration_model": {
                "env": {
                    "__doc__": "env settings, typically are specificed by the external package, including the mode for solver or for smoother",
                    "get_momentum_from_external_network": "use external network to predict momentum, notice that the momentum network is not built in this package",
                    "reg_factor": "regularzation factor",
                    "use_ode_tuple": "once use torchdiffeq package, take the tuple input or tensor input",
                    "use_odeint": "using torchdiffeq package as the ode solver"
                },
                "forward_model": {
                    "smoother": {
                        "multi_gaussian_stds": "std deviations for the Gaussians",
                        "multi_gaussian_weights": "weights for the multiple Gaussians",
                        "type": "type of smoother (diffusion|gaussian|adaptive_gaussian|multiGaussian|adaptive_multiGaussian|gaussianSpatial|adaptiveNet)"
                    }
                },
                "loss": {
                    "__doc__": "settings for the loss function",
                    "display_max_displacement": "displays the current maximal displacement",
                    "limit_displacement": "[True/False] if set to true limits the maximal displacement based on the max_displacement_setting",
                    "max_displacement": "Max displacement penalty added to loss function of limit_displacement set to True"
                },
                "shooting_vector_momentum": {
                    "__doc__": "settings for shooting vector momentum methods",
                    "use_velocity_mask_on_boundary": "a mask to force boundary velocity be zero, the value of the mask is from 0-1"
                },
                "similarity_measure": {},
                "spline_order": "Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline",
                "type": "Name of the registration model",
                "use_CFL_clamping": "If the model uses time integration, CFL clamping is used"
            }
        }
    }





**Mermaid settings on RDMM**

..   code::

    {
        "model": {
            "deformation": {
                "compute_similarity_measure_at_low_res": "to compute Sim at lower resolution"
            },
            "registration_model": {
                "env": {
                    "__doc__": "env settings, typically are specificed by the external package, including the mode for solver or for smoother",
                    "addition_smoother": "using torchdiffeq package as the ode solver",
                    "get_momentum_from_external_network": "use external network to predict momentum, notice that the momentum network is not built in this package",
                    "get_preweight_from_network": "deploy network to predict preweights of the smoothers",
                    "reg_factor": "regularzation factor",
                    "use_ode_tuple": "once use torchdiffeq package, take the tuple input or tensor input",
                    "use_odeint": "using torchdiffeq package as the ode solver"
                },
                "forward_model": {
                    "smoother": {
                        "clamp_local_weight": "clmap the preweight predicted by the network",
                        "deep_smoother": {
                            "deep_network_local_weight_smoothing": "0.02 prefered,How much to smooth the local weights (implemented by smoothing the resulting velocity field) to assure sufficient regularity",
                            "diffusion_weight_penalty": "Penalized the squared gradient of the weights",
                            "edge_penalty_filename": "Edge penalty image",
                            "edge_penalty_gamma": "Constant for edge penalty: 1.0/(1.0+gamma*||\\nabla I||*min(spacing)",
                            "edge_penalty_terminate_after_writing": "Terminates the program after the edge file has been written; otherwise file may be constantly overwritten",
                            "edge_penalty_write_to_file": "If set to True the edge penalty is written into a file so it can be debugged",
                            "estimate_around_global_weights": "If true, a weighted softmax is used so the default output (for input zero) are the global weights",
                            "network_penalty": "factor by which the L2 norm of network weights is penalized",
                            "normalization_type": "Normalization type between layers: ['batch'|'layer'|'instance'|'group'|'none']",
                            "normalize_last_layer": "If set to true normalization is also used for the last layer",
                            "normalize_last_layer_initial_affine_slope": "initial slope of affine transformation for batch and group normalization",
                            "normalize_last_layer_type": "Normalization type between layers: ['batch'|'layer'|'instance'|'group'|'none']",
                            "randomly_initialize_network": "Randomly initialize the network weights",
                            "smooth_image_for_edge_detection": "Smooth image for edge detection",
                            "smooth_image_for_edge_detection_std": "Standard deviation for edge detection",
                            "standardize_display_standardization": "Outputs statistical values before and after standardization",
                            "standardize_divide_input_images": "Value to divide the input images by *AFTER* subtraction",
                            "standardize_divide_input_momentum": "Value to divide the input momentum by *AFTER* subtraction",
                            "standardize_input_images": "if true, subtracts the value specified by standardize_subtract_from_input_images followed by division by standardize_divide_input_images from all input images to the network",
                            "standardize_input_momentum": "if true, subtracts the value specified by standardize_subtract_from_input_momentum followed by division by standardize_divide_input_momentum from the input momentum to the network",
                            "standardize_subtract_from_input_images": "Subtracts this value from all images input into a network",
                            "standardize_subtract_from_input_momentum": "Subtracts this value from the input momentum into a network",
                            "total_variation_weight_penalty": "Penalize the total variation of the weights if desired",
                            "type": "type of deep smoother (simple_consistent|encoder_decoder|clustered|simple_unet|unet|unet_no_skip)",
                            "use_current_image_as_input": "If true, uses current image as input",
                            "use_momentum_as_input": "If true, uses the image and the momentum as input",
                            "use_noise_layers": "If set to true noise is injected before the nonlinear activation function and *after* potential normalization",
                            "use_noisy_convolution": "when true then the convolution layers will be replaced by noisy convolution layer",
                            "use_source_image_as_input": "If true, uses the source image as additional input",
                            "use_target_image_as_input": "If true, uses the target image as additional input",
                            "weight_range_factor": "the factor control the change of the penality ",
                            "weight_range_init_weight_penalty": "Penalize to the range of the weights",
                            "weighting_type": "Type of weighting: w_K|w_K_w|sqrt_w_K_sqrt_w"
                        },
                        "evaluate_but_do_not_optimize_over_shared_registration_parameters": "If set to true then shared registration parameters (e.g., the network or global weights) are evaluated (should have been loaded from a previously computed optimized state), but are not being optimized over",
                        "freeze_parameters": "if set to true then all the parameters that are optimized over are frozen (but they still influence the optimization indirectly; they just do not change themselves)",
                        "gaussian_std_min": "minimal allowed std for the Gaussians",
                        "gaussian_weight_min": "minimal allowed weight for the Gaussians",
                        "load_dnn_parameters_from_this_file": "If not empty, this is the file the DNN parameters are read from; useful to run a pre-initialized network",
                        "local_pre_weight_max": "max  weight  allowed for the preweight",
                        "multi_gaussian_stds": "std deviations for the Gaussians",
                        "multi_gaussian_weights": "weights for the Gaussians std",
                        "omt_power": "Power for the optimal mass transport (i.e., to which power distances are penalized",
                        "omt_use_log_transformed_std": "If set to true the standard deviations are log transformed for the computation of OMT",
                        "omt_weight_penalty": "Penalty for the optimal mass transport",
                        "optimize_over_deep_network": "if set to true the smoother will optimize over the deep network parameters; otherwise will ignore the deep network",
                        "optimize_over_smoother_stds": "if set to true the smoother will optimize over standard deviations",
                        "optimize_over_smoother_weights": "if set to true the smoother will optimize over the *global* weights",
                        "preweight_input_range_weight_penalty": "Penalty for the input to the preweight computation; weights should be between 0 and 1. If they are not they get quadratically penalized; use this with weighted_linear_softmax only.",
                        "scale_global_parameters": "If set to True the global parameters are scaled for the global parameters, to make sure energies decay similarly as for the deep-network weight estimation",
                        "start_optimize_over_nn_smoother_parameters_at_iteration": "Does not optimize the nn smoother parameters before this iteration",
                        "start_optimize_over_smoother_parameters_at_iteration": "Does not optimize the parameters before this iteration",
                        "type": "type of smoother (diffusion|gaussian|adaptive_gaussian|multiGaussian|adaptive_multiGaussian|gaussianSpatial|adaptiveNet)",
                        "use_multi_gaussian_regularization": "If set to true then the regularization for w_K_w or sqrt_w_K_sqrt_w will use multi-Gaussian smoothing (not the velocity) of the deep smoother",
                        "use_weighted_linear_softmax": "If set to ture use the use_weighted_linear_softmax to compute the pre-weights, otherwise use stable softmax"
                    }
                },
                "load_velocity_from_forward_model": "load_velocity_from_forward_model",
                "loss": {
                    "__doc__": "settings for the loss function",
                    "display_max_displacement": "displays the current maximal displacement",
                    "limit_displacement": "[True/False] if set to true limits the maximal displacement based on the max_displacement_setting",
                    "max_displacement": "Max displacement penalty added to loss function of limit_displacement set to True"
                },
                "shooting_vector_momentum": {
                    "__doc__": "settings for shooting vector momentum methods",
                    "adapt_model": {
                        "__doc__": "settings for adaptive smoothers",
                        "clamp_local_weight": "true:clamp the local weight",
                        "compute_on_initial_map": "true:  compute the map based on initial map, false: compute the map based on id map first, then interp with the initial map",
                        "local_pre_weight_max": "clamp the value from -value to value",
                        "update_sm_by_advect": "true: advect smoother parameter for each time step  false: deploy network to predict smoother params at each time step",
                        "update_sm_with_interpolation": "true: during advection, interpolate the smoother params with current map  false: compute the smoother params by advection equation",
                        "use_predefined_weight": "use predefined weight for adapt smoother"
                    },
                    "use_velocity_mask_on_boundary": "a mask to force boundary velocity be zero, the value of the mask is from 0-1"
                },
                "similarity_measure": {},
                "spline_order": "Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline",
                "type": "Name of the registration model",
                "use_CFL_clamping": "If the model uses time integration, CFL clamping is used"
            }
        }
    }

