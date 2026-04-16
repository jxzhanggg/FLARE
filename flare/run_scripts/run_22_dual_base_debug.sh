
cd ..



source ~/.bashrc
conda activate av2vec


export OPENBLAS_NUM_THREADS=1
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


expdir="/train8/asrprg/anonymous/exp/chime_lippretrain/exp_22_base_debug"
data="$PWD/data_demo/tsv_merge_face_lip_dataset"
noise="$PWD/data_demo/tsv_noise"


python hydra_train.py --config-dir conf/pretrain \
    --config-name base_whisper_distill_offline_bestrq \
    common.tensorboard_logdir=$expdir \
    task.data=$data \
    task.label_dir=$data \
    hydra.run.dir=$expdir \
    task.noise_wav=$noise \
    dataset.max_tokens=600 \
    model.masking_type="input2" \
    checkpoint.reset_dataloader=true \
    +model.frontend="convnet" \
    +model.encoder_type="qwenconv" \
    +model.encoder_embed_dim=512 \
    +model.encoder_layers=12 \
    +model.encoder_attention_heads=8 \
    +model.num_key_value_heads=2 \
    model.modality_dropout=0.75 \
    model.audio_dropout=1.0 \
    optimization.update_freq=[1] \
    distributed_training.distributed_world_size=1 \
    optimization.max_update=800010 \
    checkpoint.save_interval_updates=10000 \
    checkpoint.keep_interval_updates=4 \
    optimization.lr=[0.0005] \
    common.log_interval=1 \
    common.user_dir=`pwd` 
