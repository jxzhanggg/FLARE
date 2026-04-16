ckpt="/train8/asrprg/anonymous/exp/chime_lippretrain/exp_21/checkpoints/checkpoint_last.pt"

data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_vox_data/tsv_lrs3_para_30h"
bash run_finetune_fn_cfg.sh $ckpt v $data base_lrs3_30h 30h_face_lip_600k


data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_vox_data/tsv_lrs3_para_30h/face"
bash run_finetune_fn_cfg.sh $ckpt v $data base_lrs3_30h 30h_face_600k


data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_vox_data/tsv_lrs3_para_30h/lip"
bash run_finetune_fn_cfg.sh $ckpt v $data base_lrs3_30h 30h_lip_600k


data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_vox_data/tsv_lrs3_para"
bash run_finetune_fn_cfg.sh $ckpt v $data base_lrs3_433h 433h_face_lip_600k

data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_vox_data/tsv_lrs3_para/face"
bash run_finetune_fn_cfg.sh $ckpt v $data base_lrs3_433h 433h_face_600k

data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_vox_data/tsv_lrs3_para/lip"
bash run_finetune_fn_cfg.sh $ckpt v $data base_lrs3_433h 433h_lip_600k


