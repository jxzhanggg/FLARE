

export PYTHONPATH=$PWD:$PYTHONPATH

# data mainfest path
data="/train8/asrprg/anonymous/avsr_data_en/WildVSR/tsv_para"
# checkpoint path
ckpt="$PWD/pretrained_models/lipreading_exp_4_base_face_lip_s2/checkpoints/checkpoint_4_60000.pt"


bash decode_scripts/decode_noise_s3_chime.sh $ckpt v 1 $data wildvsr_lip_debug