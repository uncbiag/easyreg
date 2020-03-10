cd /playpen-raid/zyshen/reg_clean/demo
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p0.txt   -o=/playpen-raid/zyshen/data/oai_seg/atlas/train_p0 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p1.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas/train_p1 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p2.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas/train_p2 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai -txt=/playpen-raid/zyshen/data/oai_seg/atlas/p3.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas/train_p3 -g=0 &
python demo_for_easyreg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm_oai -txt=/playpen-raid/zyshen/data/oai_seg/atlas/test.txt    -o=/playpen-raid/zyshen/data/oai_seg/atlas/test -g=0 &
