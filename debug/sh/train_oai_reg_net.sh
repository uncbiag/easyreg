

python gen_aug_samples_with_affine.py --txt_path=/playpen-raid1/zyshen/data/reg_oai_aug/momentum_lresol.txt --output_path=/playpen-raid1/zyshen/data/reg_oai_aug/data_aug_sr --mermaid_setting_path=/playpen-raid1/zyshen/data/reg_oai_aug/train_lddmm_momentum/reg/res/records/nonp_setting.json &
python gen_aug_samples_with_affine.py --txt_path=/playpen-raid1/zyshen/data/reg_oai_aug/momentum_lresol.txt --output_path=/playpen-raid1/zyshen/data/reg_oai_aug/data_aug_sr --mermaid_setting_path=/playpen-raid1/zyshen/data/reg_oai_aug/train_lddmm_momentum/reg/res/records/nonp_setting.json &
python gen_aug_samples_with_affine.py --txt_path=/playpen-raid1/zyshen/data/reg_oai_aug/momentum_lresol.txt --output_path=/playpen-raid1/zyshen/data/reg_oai_aug/data_aug_sr --mermaid_setting_path=/playpen-raid1/zyshen/data/reg_oai_aug/train_lddmm_momentum/reg/res/records/nonp_setting.json &
python demo_for_easyreg_train.py -o /playpen-raid1/zyshen/data -dtn=reg_oai_aug -tn=svf_net -ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_train -g=0
python demo_for_easyreg_train.py -o /playpen-raid1/zyshen/data -dtn=reg_oai_aug -tn=svf_net_scratch -ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_train_scratch -g=0
python demo_for_easyreg_train.py -o /playpen-raid1/zyshen/data/reg_oai_aug -dtn=aug_net_bspline -tn=seg -ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_train_bspline -g=0

python demo_for_easyreg_train.py -o /playpen-raid1/zyshen/data/reg_oai_aug -dtn=svf_aug -tn=svf_lld -ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_train_aug  -g=0
python demo_for_easyreg_train.py -o /playpen-raid1/zyshen/data/reg_oai_aug -dtn=svf_aug -tn=svf_lld_scratch -ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_train_aug_scratch  -g=2



-ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_test_aug_scratch -txt=/playpen-raid1/zyshen/data/reg_oai_aug/test/pair_path_list.txt   -o=/playpen-raid1/zyshen/data/reg_oai_aug/svf_lld_scratch_res -g=3
-ts=/playpen-raid/zyshen/reg_clean/debug_reg/oai_reg_test_scratch -txt=/playpen-raid1/zyshen/data/reg_oai_aug/test/pair_path_list.txt   -o=/playpen-raid1/zyshen/data/reg_oai_aug/svf_net_scratch_res -g=3