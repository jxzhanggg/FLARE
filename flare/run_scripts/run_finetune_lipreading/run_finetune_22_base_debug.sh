



ckpt="/train8/asrprg/anonymous/exp/chime_lippretrain/exp_22_base/checkpoints/checkpoint_last.pt"
data="$PWD/data_demo/tsv_merge_face_lip_dataset"
subexp="v/face_lip_4966h_cfg1_debug"



cd ../../


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



fullexpdir=$ckpt
expdir=$(dirname "$fullexpdir")
expdir=$(dirname "$expdir")



modalities=["video"]

python hydra_train.py --config-dir conf/finetune \
    --config-name base_lrs3_433h.yaml \
    common.tensorboard_logdir=$expdir/finetuned/$subexp \
    task.data=$data \
    task.label_dir=$data \
    task.tokenizer_bpe_model=$PWD/data_demo/bpe_model/spm_unigram1000.model \
    dataset.num_workers=3 \
    optimization.max_update=180000 \
    lr_scheduler.warmup_steps=60000 \
    lr_scheduler.decay_steps=120000 \
    model.freeze_finetune_updates=90000 \
    dataset.max_tokens=600 \
    hydra.run.dir=$expdir/finetuned/$subexp \
    model.w2v_path=$fullexpdir \
    task.modalities=$modalities \
    common.log_interval=1 \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq=[1] \
    common.user_dir=`pwd`