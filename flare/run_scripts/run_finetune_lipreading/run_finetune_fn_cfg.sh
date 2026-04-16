

cd ../../


source ~/.bashrc
conda activate av2vec

export OPENBLAS_NUM_THREADS=1
# export NCCL_SOCKET_IFNAME=eno2.100 #if you use single machine, comment this
export NCCL_DEBUG_SUBSYS=P2P,SHM,NET
export NCCL_IB_GID_INDEX=3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_P2P_LEVEL=NVL
export NCCL_LL_THRESHOLD=0
export OMP_NUM_THREADS=1


export HYDRA_FULL_ERROR=1

# which python
# export CUDA_VISIBLE_DEVICES=1,2
# export MASTER_ADDR="localhost"
# export MASTER_PORT='23456'
# export WORLD_SIZE=1
# export RANK=0

fullexpdir=$1
input=$2
data=$3
cfg=$4
name=$5

expdir=$(dirname "$fullexpdir")
expdir=$(dirname "$expdir")

# echo "$expdir"
# expdir=/train8/asrprg/anonymous/exp/av2vec_mlm/0_fea_reg_1_ffn_in_avg8
# ckpt=checkpoints/checkpoint_142_200000.pt



if [ $input = 'a' ]; then
    modalities=["audio"]
    subexp="finetuned/audio_$name"

elif [ $input = 'v' ]; then
    modalities=["video"]
    subexp="finetuned/video_$name"

elif [ $input = "av" ]; then
    modalities=["audio","video"]
    subexp="finetuned/audiovideo_$name"
else
    echo  "modality must be a, v or av"
    exit
fi

echo $modalities

python hydra_train.py --config-dir conf/finetune \
    --config-name $cfg \
    common.tensorboard_logdir=$expdir/$subexp \
    task.data=$data \
    task.label_dir=$data \
    task.tokenizer_bpe_model=/train8/asrprg/anonymous/avsr_data_en/LRS3/spm1000/spm_unigram1000.model \
    dataset.num_workers=3 \
    hydra.run.dir=$expdir/$subexp \
    model.w2v_path=$fullexpdir \
    task.modalities=$modalities \
    common.log_interval=10 \
    distributed_training.distributed_world_size=8 \
    optimization.update_freq=[1] \
    common.user_dir=`pwd`