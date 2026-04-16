

export CUDA_VISIBLE_DEVICES=0
source ~/.bashrc
conda activate av2vec
ckpt="$PWD/pretrained_models/exp_22_base/checkpoint_last.pt"


data="$PWD/data_demo/tsv_merge_face_lip_dataset"
results_path=$(dirname $(dirname $ckpt))/decode/lrs3_face_lip_debug

python infer_s2s.py \
    --config-dir ./conf/ --config-name s2s_decode.yaml \
    dataset.gen_subset=test \
    common_eval.path=$ckpt \
    common_eval.results_path=$results_path \
    override.data=$data \
    dataset.batch_size=1 \
    override.modalities=["video"] \
    override.label_dir=$data \
    common.user_dir=`pwd` \


