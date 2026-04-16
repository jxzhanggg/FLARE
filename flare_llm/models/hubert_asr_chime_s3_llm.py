# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional, Dict, List, Tuple, Union

import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
import torch.nn.functional as F


from transformers import AutoModel, AutoConfig, AutoTokenizer
from ..qwen_7b_base.modeling_qwen2 import Qwen2ForCausalLM

from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig
import random

from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder


DBG=True if len(sys.argv) == 1 else False

if DBG:

    from decoder import TransformerDecoder
    from utils import compute_mask_indices
else:

    from .decoder import TransformerDecoder
    from .utils import compute_mask_indices

logger = logging.getLogger(__name__)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


@dataclass
class AVHubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None
    max_update: int = II("optimization.max_update")


class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model, lora_visual=False):
        super().__init__(None)

        if lora_visual:
            lora_config = LoraConfig(
            task_type=None, 
            inference_mode=False,
            r=128,
            lora_alpha=256, 
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])

            self.w2v_model = get_peft_model(w2v_model, lora_config)
            self.w2v_model.print_trainable_parameters()

        else:
            self.w2v_model = w2v_model

    def forward(self, source, padding_mask, **kwargs):


        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask}
    

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out




class CTCFFN(nn.Module):
    def __init__(self, input_dim, voc_size):
        super(CTCFFN, self).__init__()
        self.ffn1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(1024, voc_size+1)
        self.voc_size = voc_size
    
    def forward(self, x, lengths, label, label_lengths):
        x = self.ffn1(x)
        x = self.relu(x)
        x = self.ffn2(x)
        ctc_out = x
        lprob = F.log_softmax(ctc_out, dim=2)
        blank = self.voc_size
        loss = F.ctc_loss(lprob.float(), label, lengths, label_lengths, blank=blank, zero_infinity=True)
        return ctc_out, loss, lprob



@dataclass
class AVHubertSeq2SeqLLMConfig(AVHubertAsrConfig):
    tf_mask: bool = False
    tf_mask_adaptive_tmask: bool = False
    llm_finetune: bool = False
    # llm_ft_fix: int = 0
    pretrained_path: str = ''
    smoothed_loss: bool = True
    mask_label_prob: float = 0.25

    kv_dim: int = 768 
    downsample: bool = False

    use_lora: bool = True
    pretrained_encoder_path: str = '' 
    lora_ffn: bool = False
    llm_14b: bool = False  




@register_model("av_hubert_seq2seq_chime_s3_llm", dataclass=AVHubertSeq2SeqLLMConfig)
class AVHubertSeq2SeqLLM_chime_s3_llm(BaseFairseqModel):
    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder

        self.cfg = cfg
        self.tf_mask = cfg.tf_mask

        
        if cfg.downsample:
            input_dim = cfg.kv_dim * 2
        else:
            input_dim = cfg.kv_dim
        
        if cfg.llm_14b:
            self.modality_projection = nn.Sequential(
                                        nn.Linear(input_dim, 2048),
                                        nn.GELU(),
                                        nn.Linear(2048, 5120),
                                        nn.LayerNorm(5120))
            # your hugging face downloaded qwen 2.5 14B base weights path
            model_name_or_path = "pretrain_models/qwen2.5_14B"
        else:

            self.modality_projection = nn.Sequential(
                                        nn.Linear(input_dim, 2048),
                                        nn.GELU(),
                                        nn.Linear(2048, 3584),
                                        nn.LayerNorm(3584))
            # your hugging face downloaded qwen 2.5 7B base weights path
            model_name_or_path = "pretrain_models/QW2.5"
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.llm = Qwen2ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)


        self.use_lora = self.cfg.use_lora

        if self.cfg.lora_ffn:
            target_modules = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]

        if self.use_lora:
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.1

            lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            target_modules=target_modules)

            
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        else:
            for name, params in self.llm.named_parameters():
                params.requires_grad = False

        self.smoothed_loss = cfg.smoothed_loss

        self.mask_id = 151665 # unused one 
        self.pad_id = self.llm_tokenizer.eos_token_id
        #         
        if cfg.pretrained_path != "":
            loaded_ckpt = torch.load(cfg.pretrained_path)
            self.load_state_dict(loaded_ckpt["model"], strict=False)
            logger.info(f'loading from pretrained path {cfg.pretrained_path}.')

        if cfg.max_update <= cfg.freeze_finetune_updates:
            self.save_encoder = False
        else:
            self.save_encoder = True


    def get_my_state_dict(self,):
        
        save_model_dict = {}
        if self.use_lora:
            save_model_dict.update({"llm." + k: v.to("cpu") for k, v in self.llm.named_parameters() if v.requires_grad})
        
        save_model_dict.update({"modality_projection."+k:v for k, v in self.modality_projection.state_dict().items()})

        if self.save_encoder:
            save_model_dict.update({"encoder." +k:v for k, v in self.encoder.state_dict().items()})

        return save_model_dict


    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # 递归地重写state中的cfg，所以实现了覆盖。
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,}


        # print('encoder_layerdrop overrides -> ', cfg.layerdrop) #0.1
        if cfg.w2v_args is None:
            # overwrite_args_by_name， 递归地重写cfg，实现了参数覆盖。
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])
        
        
        # hard code here, maybe should be fixed.
        w2v_args.model._name = "av_hubert_distill_whipser_offline_bestrq_dual_frontend"
        

        encoder_ = task_pretrain.build_model(w2v_args.model)
        encoder = HubertEncoderWrapper(encoder_,)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)
        else:
            assert False

        del state

        if cfg.pretrained_encoder_path != '':
            loaded_ckpt = torch.load(cfg.pretrained_encoder_path)
            state_dict = {}
            for k, v in loaded_ckpt["model"].items():
                if k.startswith('encoder.w2v_model.'):
                    new_k = k.replace('encoder.w2v_model.', '') 
                    # print(k, '->', new_k)
                    state_dict[new_k] = v
            encoder.w2v_model.load_state_dict(state_dict, strict=True)
            logger.info(f'Load encoder from {cfg.pretrained_encoder_path} successfully.')

        encoder.w2v_model.remove_pretraining_modules()
        


        if cfg.max_update <= cfg.freeze_finetune_updates:
            for name, params in encoder.named_parameters():
                params.requires_grad = False
            logger.info(f'max update {cfg.max_update} <= freeze update {cfg.freeze_finetune_updates}. encoder is freezed.')


        
        return AVHubertSeq2SeqLLM_chime_s3_llm(encoder, cfg)
    

    def forward(self, **kwargs):


            
        if kwargs["source"].get("audio", None) is not None:
            device = kwargs["source"]['audio'].device
        elif kwargs["source"]['video'] is not None:
            face, lip = kwargs["source"]['video']
            if face is not None:
                device = face.device
            else:
                device = lip.device
        
        if  (self.training and self.tf_mask):

            video_face, video_lip = kwargs["source"]['video'] # [B, C, T, H, W]

            if random.random() < 0.5:

                def _mask(video):
                    n_tmask = max(1, int(video.shape[2] // 25.))

                    for i in range(n_tmask):
                        t = random.randint(0, 10)
                        t = min(t, int(video.shape[2]*0.5))
                        t0 = random.randint(0, video.shape[2] - t)
                        video[:,:, t0:t0+t] = 0.
                    return video
                
                if video_face is not None:
                    video_face = video_face.clone()
                    video_face = _mask(video_face)
                
                if video_lip is not None:
                    video_lip = video_lip.clone()
                    video_lip = _mask(video_lip)

                kwargs["source"]['video'] =  (video_face, video_lip)
            
            n_fmask = 2
            n_tmask = 2
            n_filter = 104
            if kwargs["source"]['audio'] is not None:
                audio = kwargs["source"]['audio'].clone() # [B, F, T]
                if random.random() < 0.5:
                    for i in range(n_fmask):
                        f = random.randint(0, 12)
                        f0 = random.randint(0, n_filter - f)
                        audio[:, f0:f0+f, :] = 0
                    
                    for i in range(n_tmask):
                        t = random.randint(0, 10)
                        t = min(t, int(audio.shape[-1]*0.4))
                        t0 = random.randint(0, audio.shape[-1] - t)
                        audio[:, :, t0:t0+t] = 0

        ft =  False if (self.cfg.freeze_finetune_updates > self.num_updates) else True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output_encoder = self.encoder.forward(**kwargs)
 

        hidden_audio = output_encoder["encoder_out"] # [T, B, D]
        padding_mask = output_encoder["encoder_padding_mask"] #[B, T] True if mask

        hidden_len = (~padding_mask).long().sum(dim=1)    
        prompt_token = self.get_prompt(drop_audio=False, 
                                    drop_video=False, 
                                    device=device)
        def _flatten_sents(hidden, hidden_length):
            # hidden : [T, B, D], hidden_length: [B, T]
            hidden_list = []
            for batch_idx in range(hidden.shape[1]):
                cutted_hidden = hidden[:hidden_length[batch_idx], batch_idx, :]
                hidden_list.append(cutted_hidden)
            return hidden_list
        

        if self.cfg.downsample:
            # hidden_audio: (T, B, D)
            T, B, D = hidden_audio.shape
            # 如果 T 是奇数，右侧补一帧（补零）
            if T % 2 == 1:
                hidden_audio = F.pad(hidden_audio, (0, 0, 0, 0, 0, 1))  # pad T dimension
                T += 1
            # even: (T/2, B, D) 取 0,2,4...
            even = hidden_audio[0:T:2]
            # odd:  (T/2, B, D) 取 1,3,5...
            odd  = hidden_audio[1:T:2]
            # 在 feature/channel 维拼接: (T/2, B, 2D)，格式 [even ; odd]
            hidden_audio = torch.cat([even, odd], dim=-1)
            # 长度更新：ceil(len/2)
            hidden_len = (hidden_len + 1) // 2
    
        projected_hidden_audio = self.modality_projection(hidden_audio) # t, b, d
        hidden_list_audio = _flatten_sents(projected_hidden_audio, hidden_len)
        hidden_list = hidden_list_audio

        target_list = kwargs["target"]
        batch_size = len(target_list)
        prompt_token = prompt_token.unsqueeze(0).expand(batch_size, -1) # [b, N]

        A_token = self.llm_tokenizer.encode("A")

        if self.use_lora:
            QWEN = self.llm.model.model
        else:
            QWEN = self.llm.model

        A_embedding = QWEN.embed_tokens(torch.tensor(A_token).to(device))

        concat_lengths = [len(h) + len(A_token) +  len(prompt_token[i]) + len(t) for i, (h, t) in enumerate(zip(hidden_list, target_list))]
        max_concat_lengths = max(concat_lengths)
        
        input_ids, target_ids =[],  []
        input_embeddings = []
        for batch_idx,  target_token in enumerate(target_list):

            if self.cfg.mask_label_prob > 0 and self.training:
                prob_mat = torch.concat([torch.zeros(1),
                                        torch.full((len(target_token) - 1,), self.cfg.mask_label_prob)])

                masked_indices = torch.bernoulli(prob_mat).bool().to(target_token.device)
                masked_target_token = target_token.clone()
                masked_target_token[masked_indices] = self.mask_id
            
            else:
                masked_target_token = target_token.clone()
        
            padded_len = max_concat_lengths - concat_lengths[batch_idx]
            projected_speech_embedding = torch.concat([A_embedding,  hidden_list_audio[batch_idx]], dim=0)
            llm_token_embedding = QWEN.embed_tokens(masked_target_token) # N -> N, D
            prompt_embedding = QWEN.embed_tokens(prompt_token[batch_idx]) # N' -> N', D

            if padded_len> 0:
                pad_embeddings = QWEN.embed_tokens((target_token.new_ones(padded_len) * self.pad_id).long())
                input_embedding = torch.concat([projected_speech_embedding, prompt_embedding, llm_token_embedding, pad_embeddings], dim=0) # T +N+P, D
            else:
                input_embedding =  torch.concat([projected_speech_embedding, prompt_embedding, llm_token_embedding], dim=0) # T +N, D

            if padded_len > 0:
                target_id = torch.concat([
                    target_token.new_ones(len(projected_speech_embedding) + len(prompt_embedding) + 1)  * -100,
                    target_token[1:],
                    (target_token.new_ones(padded_len) * -100) ])
                
                input_id = torch.concat([
                    target_token.new_zeros(len(projected_speech_embedding)),
                    prompt_token[batch_idx],
                    masked_target_token,
                    target_token.new_zeros(padded_len)])

            else:
                target_id = torch.concat([
                    target_token.new_ones(len(projected_speech_embedding) +  len(prompt_embedding) + 1) * -100,
                    target_token[1:]])
                
                input_id = torch.concat([
                    target_token.new_zeros(len(projected_speech_embedding)),
                    prompt_token[batch_idx],
                    masked_target_token])
            
            target_ids.append(target_id)
            input_ids.append(input_id)
            input_embeddings.append(input_embedding)

        target_ids = torch.stack(target_ids).contiguous() # b, t
        fake_input_ids = torch.stack(input_ids).contiguous()
        input_embeddings = torch.stack(input_embeddings).contiguous() # b, t, t

        
        (llm_outputs, acc) = self.llm.forward(
                inputs_embeds = input_embeddings, # .transpose(0, 1).contiguous(), # b, t, d -> t, b, d
            labels = target_ids)

        loss_ce = llm_outputs.loss


        results = {}
               
        results["sample_size"] = (target_ids != -100).long().sum()

        results["losses"] = {
            "loss_ce" : loss_ce}
            
        results["accuracy"] = acc

        return results
    

    def get_prompt(self, drop_audio, drop_video, device, set_prompt=False):
        if drop_audio:
            prompt = "transcribe the video: "
        elif drop_video:
            prompt = "transcribe the audio: "
        else:
            prompt = "transcribe the audio and video: "
        prompt_token = self.llm_tokenizer.encode(prompt)
        prompt_token = torch.tensor(prompt_token).to(device)
        if set_prompt:
            self.prompt_str = prompt
        return prompt_token


    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)

        self.num_updates = num_updates



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
