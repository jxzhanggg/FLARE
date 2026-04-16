# cd ..
ckpt=$1
modality=$2
export CUDA_VISIBLE_DEVICES=$3
# noise=$4
data=$4
expname=$5


source ~/.bashrc
conda activate av2vec
module unload cuda/8.0
module load cuda/11.7.1-cudnn-v8.5.0

#ckpt="/train8/asrprg/anonymous/exp/llm-vsr/s1_1_lr2e-4_fz22k/checkpoints/checkpoint_last.pt"



if [ $modality = "a" ]; then
    modalities=["audio"]
elif [ $modality = 'v' ]; then
    modalities=["video"]
elif [ $modality = 'av' ]; then
    modalities=["audio","video"]
else
    echo "must be a, v or av"
    exit
fi

export HYDRA_FULL_ERROR=1

temp=1.2

#if [ $noise = 'true' ]; then
#echo "Decode noise"

results_path=$(dirname $ckpt)
results_path=$(dirname $results_path)/decode_bs30/$modality/$expname

echo $PWD


# 这时python path 会进入decode_scripts/
# python infer_s2s_beamsearch_sep_adaptor.py \
# data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/data_tsv"
# data="/train8/asrprg/anonymous/avsr_data_en/LRS3/test_set_avsr/all_snr"

python  decode_scripts/infer_s2s_beamsearch_s3_chime_llm.py \
    --config-dir conf --config-name s2s_decode.yaml \
    dataset.gen_subset=test \
    dataset.batch_size=1 \
    common_eval.path=$ckpt \
    generation.beam=30 \
    temp=$temp \
    common_eval.results_path=$results_path \
    override.data=$data \
    override.modalities=$modalities \
    override.label_dir=$data \
    common.user_dir=`pwd` \



#  dataset.batch_size=1 \
# # generation.beam=50 \
