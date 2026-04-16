

# FLARE-LLM finetuning and decoding


## LLM-Finetuned weights (LoRA and adaptor weights are saved)

Download and unzip the pretrained_models directory
- Large stage3 pretrained_models/lipreading_exp_2_large_face_lip_s2/checkpoints/checkpoint_4_60000.pt
- Base stage3 pretrained_models/lipreading_exp_4_base_face_lip_s2/checkpoints/checkpoint_4_60000.pt


## Finetune scripts:

- Large
stage2 (finetune adaptor): run_scripts/run_lipreading/run_lipreading_exp_2_large_face_lip_s1.sh
stage3 (finetune lora+adaptor): run_scripts/run_lipreading/run_lipreading_exp_2_large_face_lip_s2.sh

- Base
stage2 (finetune adaptor): run_scripts/run_lipreading/run_lipreading_exp_4_base_face_lip_s1.sh
stage3 (finetune lora+adaptor): run_scripts/run_lipreading/run_lipreading_exp_4_base_face_lip_s2.sh

- DEBUG demo
run_scripts/run_lipreading/run_lipreading_exp_4_base_face_lip_s2_debug.sh


## Decoding scripts:

- run_decode.sh


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
See ../flare/data_demo/tsv_merge_face_lip_dataset/*.tsv

- wrd file: each (i+1)-th line corresponds to the i-th line of tsv file's text transcript.
See ../flare/data_demo/tsv_merge_face_lip_dataset/*.wrd

- dict file: data_demo/dict.km.txt is not actually used, it's a legacy to be deleted. 
../flare/data_demo/dict.wrd.txt is the bpe dictionary file.
../flare/data_demo/bpe_model is the bpe tokenizer model used for finetuning.

- noise file (not used during finetuning): noise list file for adding random noise during pretraining, each line is a path to noise wav file. 
See the ../flare/data_demo/tsv_noise for an example. 

## Environment

See ../flare/environment.yml file for python environment

## Double-blind review

All hard code file/dir path are anonymized: the user/company names are replaced with anonymous/anonymous_company
