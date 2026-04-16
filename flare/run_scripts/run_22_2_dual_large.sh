
cd ..



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

# export CUDA_VISIBLE_DEVICES=1,2
# export MASTER_ADDR="localhost"
# export MASTER_PORT='23456'
# export WORLD_SIZE=2
# export RANK=0
# export NCCL_SOCKERT_IFNAME=eno2.100
# expdir=/train8/asrprg/anonymous/exp/av2vec_mlm/9_wavlm_1_2_vox_lr1e-3
#expdir=exp/pretrain2

expdir="/train8/asrprg/anonymous/exp/chime_lippretrain/exp_22_2_large"

data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_merge_dataset/tsv_merge_face_lip_dataset"

# data=/train8/asrprg/anonymous/avsr_data_en/LRS3/vox_lrs3_data
python hydra_train.py --config-dir conf/pretrain \
    --config-name base_whisper_distill_offline_bestrq \
    common.tensorboard_logdir=$expdir \
    task.data=$data \
    task.label_dir=$data \
    hydra.run.dir=$expdir \
    task.noise_wav=/train8/asrprg/anonymous/musan/musan/tsv/all_mixspeech \
    dataset.max_tokens=2200 \
    model.masking_type="input2" \
    +model.frontend="convnet" \
    +model.encoder_type="qwenconv" \
   +model.encoder_embed_dim=768 \
    +model.encoder_layers=18 \
    +model.encoder_attention_heads=12 \
    +model.num_key_value_heads=2 \
    model.modality_dropout=0.75 \
    model.audio_dropout=1.0 \
    optimization.update_freq=[4] \
    distributed_training.distributed_world_size=4 \
    optimization.max_update=800000 \
    checkpoint.save_interval_updates=10000 \
    checkpoint.keep_interval_updates=4 \
    optimization.lr=[0.0005] \
    common.log_interval=50 \
    common.user_dir=`pwd` 

