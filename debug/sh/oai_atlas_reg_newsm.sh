cd /playpen-raid/zyshen/reg_clean/demo
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai_new_sm -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p0.txt   -o=/playpen-raid/zyshen/data/oai_seg/atlas_sm/train_p0 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai_new_sm -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p1.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas_sm/train_p1 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai_new_sm -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p2.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas_sm/train_p2 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai_new_sm -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p3.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas_sm/train_p3 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai_new_sm -txt=/playpen-raid/zyshen/data/oai_seg/atlas/test.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas_sm/test -g=0 &
