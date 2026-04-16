


cd ../../






source ~/.bashrc
module unload cuda/8.0
module load cuda/11.7.1-cudnn-v8.5.0

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


ckpt="/train8/asrprg/anonymous/exp/chime_lippretrain/exp_22_base/checkpoints/checkpoint_51_800000.pt"
data="/work2/asrprg/anonymous/2025/CHiME-9/data_preprocess/process_merge_dataset/tsv_merge_face_lip_dataset"
expdir="/train8/asrprg/anonymous/exp/chime9/s3_llm/lipreading_exp_4_base_face_lip_s1"
pretrained=""
pretrained_encoder="/train8/asrprg/anonymous/exp/chime_lippretrain/exp_22_base/finetuned/v/face_lip_4966h_cfg1/checkpoints/checkpoint_last.pt"


python hydra_train.py --config-dir conf/av-finetune \
    --config-name llm_noise_pt_ft_433h \
    common.tensorboard_logdir=$expdir/avsr \
    task.data=$data \
    task.label_dir=$data \
    task.tokenizer_bpe_model=/train8/asrprg/anonymous/avsr_data_en/LRS3/spm1000/spm_unigram1000.model \
    task.noise_wav=/train8/asrprg/anonymous/musan/musan/tsv/all_mixspeech_v2 \
    task.noise_prob=0. \
    dataset.num_workers=3 \
    +model.downsample=False \
    optimization.max_update=10000 \
    lr_scheduler.warmup_steps=3000 \
    lr_scheduler.decay_steps=7000 \
    model.freeze_finetune_updates=10000 \
    dataset.max_tokens=500 \
    optimization.update_freq=[4] \
    optimization.lr=[0.001] \
    hydra.run.dir=$expdir \
    model._name="av_hubert_seq2seq_chime_s3_llm" \
    +model.pretrained_encoder_path=$pretrained_encoder \
    +model.use_lora=False \
    +task.encoder_type="distillav" \
    +task.llm_type="qwen" \
    common.bf16=True \
    common.fp16=False \
    +model.kv_dim=512 \
    model.w2v_path=$ckpt \
    +model.pretrained_path=$pretrained \
    task.modalities=["video"] \
    common.log_interval=20 \
    distributed_training.distributed_world_size=8 \
    common.user_dir=`pwd`

