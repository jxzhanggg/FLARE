

# FLARE pretrain & light-weight Transformer finetuning codes


## Pretrained and finetuned (light-weight Transformer decoder) weights

Download and unzip the pretrained_models directory
- Large pretrained: pretrained_models/exp_22_2_large/checkpoints/checkpoint_52_800000.pt
- Large finetuned: pretrained_models/exp_22_2_large/finetuned/checkpoints/checkpoint_last.pt
- Base pretrained: pretrained_models/exp_22_base/checkpoints/checkpoint_51_800000.pt
- Base finetund: pretrained_models/exp_22_base/finetuned/checkpoints/checkpoint_last.pt

## Pretrain scripts:

- Large 
run_scripts/run_22_2_dual_large.sh

- Base
run_scripts/run_22_dual_base.sh

- DEBUG demo
run_scripts/run_22_dual_base_debug.sh


## Finetune scripts:

- Large
run_scripts/run_finetune_lipreading/run_finetune_22_2_large.sh

- Base
run_scripts/run_finetune_lipreading/run_finetune_22_base.sh

- DEBUG demo
run_scripts/run_finetune_lipreading/run_finetune_22_base_debug.sh


## Decoding scripts:

- decode.sh


## Data preparation

### data types
- audio data: 16k Hz 16bit *.wav single channel
- video data: 96*96 pixel 25 fps mp4 files, for both face and lip video. The face video use LRS2/LRS3/Voxceleb style,
lip video follows process of AVHuBERT or Auto-AVSR.
- text data: lower case normlized text transcripts
- whisper features (only for pretraining): *.npy file with shape of (N_frames, Depth) for pretraining only, should be put in the path 
```python
whisper_fn = audio_fn.replace("/audio/", '/whisper_fea/layer_last1/').replace('.wav', '.npy')
```
, where audio_fn is the corresponding audio file.

### data manifest

- tsv file: first line "/" or the root path, the other line follows the format 
```python
line = f"{uttid}\t{face_path}\t{lip_path}\t{audio_path}\t{video_frames}\t{audio_samples}\n"
```
See data_demo/tsv_merge_face_lip_dataset/*.tsv

- wrd file: each (i+1)-th line corresponds to the i-th line of tsv file's text transcript.
See data_demo/tsv_merge_face_lip_dataset/*.wrd

- dict file: data_demo/dict.km.txt is not actually used, it's a legacy to be deleted. 
data_demo/dict.wrd.txt is the bpe dictionary file.
data_demo/bpe_model is the bpe tokenizer model used for finetuning.

- noise file: noise list file for adding random noise during pretraining, each line is a path to noise wav file. 
See the data_demo/tsv_noise for an example.

## Environment

See environment.yml file for python environment

## Double-blind review

All hard code file/dir path are anonymized: the user/company names are replaced with anonymous/anonymous_company
