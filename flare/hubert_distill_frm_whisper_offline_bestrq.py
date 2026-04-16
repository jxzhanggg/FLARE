# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import logging
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.linalg import vector_norm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
# from fairseq.models.wav2vec.wav2vec2 import (
#     ConvFeatureExtractionModel,
#     TransformerEncoder,
# )
from .transformer_encoder import TransformerEncoder
from .qwen_transformer import Qwen2Encoder
from .qwen_conformer import Qwen2ConvEncoder
from fairseq.modules import GradMultiply, LayerNorm
from copy import deepcopy
import math


DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from resnet import ResEncoder
    from convnext import ConvEncoder 
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from utils import compute_mask_indices
    from decoder import TransformerDecoder

else:
    from .hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from .resnet import ResEncoder
    from .convnext import ConvEncoderTT, ConvEncoderT
    from .utils import compute_mask_indices
    from .decoder import TransformerDecoder

from omegaconf import II

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    num_key_value_heads: int = field(default=2, metadata={'help': "ky heads, used only in qwen transformer."})

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
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
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})

    modality_scheduler: bool = field(default =False, metadata={'help': "use modality schedule"})
    cb_vocab: int = field(default = 8192, metadata={'help': 'cb vocab'})
    cb_dim: int = field(
        default=32,
        metadata={"help": "dropout probability for the transformer"},
    )
    encoder_type: str = field(default = "transformer", metadata={"help": "transformer, conformer"})

    frontend: str = field(default = "resnet", metadata={"help": "resnet, convnet"})


class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None
        # self.post_norm = LayerNorm(cfg.encoder_embed_dim)

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
            # print(x.shape)
        
        x = self.proj(x.transpose(1, 2))

        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x



class ApplyKmeansCUDA(torch.nn.Module):
    def __init__(self, km_path):
        super().__init__()
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose() # 2000,d -> d, 2000
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True) # 1, 2000

        C = torch.from_numpy(self.C_np)
        Cnorm = torch.from_numpy(self.Cnorm_np)

        self.C = torch.nn.Parameter(C, requires_grad=False)
        self.Cnorm = torch.nn.Parameter(Cnorm, requires_grad=False)
        

    def forward(self, x):
        # if isinstance(x, torch.Tensor):
            # x = x.to(self.C.device)
        # 't, d'
        dist = (
            x.pow(2).sum(1, keepdim=True) #t, 1
            - 2 * torch.matmul(x, self.C) # t, d x d, 2000
            + self.Cnorm  #1, 2000
        )
        return dist.argmin(dim=1), dist # t, 2000





class RandomProjectionQuantizer(nn.Module):
    """Vector quantization using a projection and a randomly initialised codebook
    this is useful for models like BEST-RQ for instance.

    The output is the indices of the closest code in the codebook for each
    time step of the input.

    ref: https://arxiv.org/pdf/2202.01855

    Arguments
    ---------
    input_dim: int
        Input dimension (channels).
    cb_dim: int
        Size of each code in the codebook.
    cb_vocab: int
        Number of codes in the codebook

    Example
    -------
    >>> quantiser = RandomProjectionQuantizer(16, 16, 32)
    >>> inputs = torch.rand(10, 12, 16)
    >>> output = quantiser(inputs)
    >>> output.shape
    torch.Size([10, 12])
    """

    def __init__(self, input_dim, cb_dim, cb_vocab):
        super().__init__()
        # pdb.set_trace()
        #160
        self.input_dim = input_dim
        #64
        self.cb_dim = cb_dim
        #8196
        self.cb_vocab = cb_vocab
        #P_init.shape =torch.Size([160, 64])
        # Section 3.1 "projection matrix A use Xavier initialization"
        P_init = torch.empty((input_dim, cb_dim))
        self.register_buffer("P", nn.init.xavier_uniform_(P_init))
        #self.P.size() =[160,64] self.register_buffer.shape = [8192,64]
        # normalize random matrix for codebook
        self.register_buffer("CB", F.normalize(torch.randn(cb_vocab, cb_dim)))

    def forward(self, x):
        """Forward the latent vector to obtain a quantised output"""
        #print("-------------->Before F.normalize(x @ self.P, dim=2): ",x )
        #pdb.set_trace()
        #x.size() [b,t,f]
        # pdb.set_trace()
        # b, t, d
        x = F.normalize(x @ self.P, dim=2)
        #x.size() [b,t,cb_dim]
        #print("-------------->After F.normalize(x @ self.P, dim=2): ",x )
        #self.CB.unsqueeze(1).size() = [8196,1,512] x.unsqueeze(1).size() [48,1,38,512]
        # (self.CB.unsqueeze(1) - x.unsqueeze(1)).size() = [48,8196,38,512]
        #pdb.set_trace()
        #vector_norm((self.CB.unsqueeze(1) - x.unsqueeze(1)), dim=-1).size() = [48,8196,38]
        #vector_norm((self.CB.unsqueeze(1) - x.unsqueeze(1)), dim=-1).argmin(dim =1).size() = [48,38]
        return vector_norm(
            (self.CB.unsqueeze(1) - x.unsqueeze(1)), dim=-1
        ).argmin(dim=1)


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape : tuple
        It is the shape of the input tensor.
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        max_norm=None,
        combine_dims=False,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]
     
        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
       
        wx = self.w(x)

        return wx


@register_model("av_hubert_distill_whipser_offline_bestrq", dataclass=AVHubertConfig)
class AVHubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: AVHubertConfig,
        task_cfg: AVHubertPretrainingConfig,
        dictionaries: List[Dictionary],
        **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers

        if cfg.frontend == 'resnet':
            resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        elif cfg.frontend == 'convnet':
            resnet = ConvEncoderT()
        elif cfg.frontend == 'convnet2':
            resnet = ConvEncoderTT()
        else:
            assert False

        num_params = sum(p.numel() for p in resnet.parameters())
        logger.info('')
        logger.info("Total number of frontend parameters: " + str(num_params))


        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video_face = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.feature_extractor_video_lip = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)

        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 3
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim


        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' or self.masking_type== 'input2' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        if self.masking_type == 'feature':
            self.mask_emb_video = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )

        if cfg.encoder_type == "transformer":
            self.encoder = TransformerEncoder(cfg)
        elif cfg.encoder_type == 'conformer':
            assert False, "deleted, not support."
        elif cfg.encoder_type == "qwen":
            self.encoder = Qwen2Encoder(cfg)
        elif cfg.encoder_type == 'qwenconv':
            self.encoder = Qwen2ConvEncoder(cfg)
        else:
            assert False
    
        self.encoder_type = cfg.encoder_type

        self.layer_norm = LayerNorm(self.embed)

        # regression
        self.linear = Linear(n_neurons=cfg.cb_vocab * 2, input_size=cfg.encoder_embed_dim, bias=True)
        quantiser_kwargs ={ "input_dim": 1280, "cb_dim": cfg.cb_dim, "cb_vocab":cfg.cb_vocab}
        self.Quantizer = RandomProjectionQuantizer(**quantiser_kwargs)


        self.final_dim = final_dim

        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_scheduler = cfg.modality_scheduler

        if task_cfg.avg == 1:
            stat_path = '/work2/asrprg/anonymous/lippretrain/extract_whisper_fea/whisper_last1.stat.npy'
        elif task_cfg.avg == 8:
            stat_path = '/work2/asrprg/anonymous/lippretrain/extract_whisper_fea/whisper_last8_avg8.stat.npy'
        
        # 转成 tensor 并注册为 buffer
        self.register_buffer("whisper_stat", torch.from_numpy(np.load(stat_path)))
        
        self.avg = task_cfg.avg


    def get_my_state_dict(self,):
        return self.state_dict()
        # save_state_dict = {}
        # for k, v in self.state_dict().items():
        #     if k.startswith('whisper.'):
        #         continue
        #     save_state_dict[k] = v
        # return save_state_dict


    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.modality_scheduler:
            ceil = 0.75
            max_anneal_updates = 150_000

            if num_updates >= max_anneal_updates:
                self.modality_dropout = ceil
            else:
                # print('update modality_dropout')
                self.modality_dropout = 0. + float(num_updates) / float(max_anneal_updates) * ceil
        


    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AVHubertConfig, task: AVHubertPretrainingTask):
        """Build a new model instance."""

        kwargs = {}
        model = AVHubertModel(cfg, task.cfg, task.dictionaries, **kwargs)
        return model
    

    def compute_bestrq_mask_indices(self, sample_lens, mask_prob: float, mask_length: int):

        L = min(sample_lens)                          # 最短时间长度
        N = int(mask_prob * L + random.random())      # 掩码段数（随机四舍五入）
        N = max(N, 1)                                 # 至少 1 段

        # 合法起点：0, M, 2M, ...；随机打乱后取前 N 个，再排序
        starts = torch.randperm(L // mask_length) * mask_length
        starts = starts[:N].sort().values

        # 展开成帧索引
        if len(starts) > 0:
            idx = torch.cat([torch.arange(s, s + mask_length) for s in starts])
            return idx  # 形状：[N * mask_length]
        else:
            return None
    

    def apply_input_mask(self, x, padding_mask):

        if isinstance(x, tuple):
            x1, x2 = x
            B, C, T = x1.shape[:3] #B, C
            is_audio = True if len(x1.shape) == 3 else False
        else:
            B, C, T = x.shape[:3] #B, C
            is_audio = True if len(x.shape) == 3 else False
        
        
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image

        sample_lens = (~padding_mask).long().sum(dim=-1)
        mask = self.compute_bestrq_mask_indices(sample_lens, mask_prob, mask_length)

        if mask is None:
            return x

        if is_audio:
            noise = torch.normal(
                    mean=0, #self.hparams.noise_mean,
                    std=0.1 ,#self.hparams.noise_std,
                    size=(B, C, mask.shape[0]),
                    device=x.device,
                    dtype=x.dtype  
                )
            if isinstance(x, tuple):
                x1[:, :, mask] = noise
                x2[:, :, mask] = noise
                x = (x1, x2)
            else:
                x[:, :, mask] = noise
        else:

            if isinstance(x, tuple):
                H, W = x1.shape[3:]
                noise = torch.normal(
                        mean=0, #self.hparams.noise_mean,
                        std=0.1 ,#self.hparams.noise_std,
                        size=(B, C, mask.shape[0], H, W),
                        device=x.device,
                        dtype=x.dtype  
                    )
                x1[:, :, mask, :, :] = noise
                x2[:, :, mask, :, :] = noise
                x = (x1, x2)

            
            else:
                H, W = x.shape[3:]
                noise = torch.normal(
                        mean=0, #self.hparams.noise_mean,
                        std=0.1 ,#self.hparams.noise_std,
                        size=(B, C, mask.shape[0], H, W),
                        device=x.device,
                        dtype=x.dtype  
                    )
                x[:, :, mask, :, :] = noise
        
        return x

    def apply_input_mask_hubert(self, x, padding_mask):
        
        tuple_x = False
        if isinstance(x, tuple):
            x1, x2 = x
            x = x1
            tuple_x = True


        B, C, T = x.shape[:3] #B, C

        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        
        if mask_prob > 0:
            # mask_indices [B, T] mask=True, unmask=False
            # starts, ends, batch_indexes: 1D arrays with start_pos, end_pos, batch_id
            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            # mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)

            x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
            if tuple_x:
                x2 = x2.transpose(1, 2).contiguous() # [B, T, C, H, W]
            # # why ?
            # if B == 1:
            #     x[mask_indices] = 0
            assert(self.selection_type == "same_seq")
            if is_audio:
                x[mask_indices] = self.mask_emb
                if tuple_x:
                    x2[mask_indices] = self.mask_emb
            
            elif self.selection_type == 'same_other_seq':
                # use clip from other batch to fill the masked range
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]

                if tuple_x:
                    x2_perm = x2[perm]
                    x2[mask_indices] = x2_perm[mask_indices]
            
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end-start
                    # ensure the other clip won't overlap with the range(start, end)
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    # clip to prevent index exceeding the max length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
                # gen indexes 
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                
                x[mask_indices] = x[batch_indexes, other_indexes]

                if tuple_x:
                    x2[mask_indices] = x2[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
            if tuple_x:
                x2 = x2.transpose(1,2).contiguous()
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            logger.info(f"No mask channel prob for input masking")
        
        if tuple_x:
            x = (x, x2)
        
        return x

    def apply_feature_mask(self, x, padding_mask, stream):

        B, T, C = x.shape
        if stream == 'audio':
            mask_emb, mask_prob, mask_length = self.mask_emb, self.mask_prob_audio, self.mask_length_audio
        elif stream == 'video':
            mask_emb, mask_prob, mask_length = self.mask_emb_video, self.mask_prob_image, self.mask_length_image
        else:
            assert False
        
        
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask
    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        # make input output the same frame rate.
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

  

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
            logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        
        elif self.sim_type == "mse":
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            feats2 = (feats_ ** 2).sum(dim=-1, keepdims=True) # t, 1
            emb_mat2 = (emb_mat ** 2).sum(dim=-1, keepdims=True) # v, 1
            mse = feats2 + emb_mat2.transpose(0, 1) - 2 * torch.matmul(feats_, emb_mat.transpose(0, 1)) # t, 1 + 1, v - 2 * t, f x f, v -> (t, v)
            logits = mse.reshape(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def enhanced_feature_drop_bft(self, features_video_face, features_video_lip, face_mask, lip_mask):
        """
        对形状为 (B, F, T) 的特征执行随机条件丢弃操作。
        使用 PyTorch 张量操作代替 Python 循环，实现 GPU 加速。
        
        Args:
            features_video_face, features_video_lip (torch.Tensor): 形状 (B, F, T)。
            face_mask, lip_mask (torch.Tensor): 形状 (B,) 的布尔或 0/1 掩码。
        """
        
        # --- 1. 识别需要随机丢弃的 Batch 索引 (B,) ---
        conditional_mask = face_mask.bool() & lip_mask.bool()
        
        # 如果没有需要操作的样本，直接返回
        if not conditional_mask.any():
            return features_video_face, features_video_lip

        # --- 2. 第一次随机采样：决定是否丢弃 (50% 概率) ---
        rand_drop = torch.rand_like(conditional_mask, dtype=torch.float)

        # do_drop_mask: 满足条件且 rand > 0.5 (即 50% 概率进行丢弃)
        do_drop_mask = conditional_mask & (rand_drop > 0.5)

        # --- 3. 第二次随机采样：决定丢弃 face 还是 lip (50% vs 50%) ---
        # 我们只需要对 do_drop_mask 为 True 的位置进行随机选择，
        # 但为了简洁和效率，在整个 B 维度上生成随机数是可接受的
        rand_choice = torch.rand_like(do_drop_mask, dtype=torch.float)

        # 最终丢弃 face 的 Batch 索引掩码
        drop_face_mask = do_drop_mask & (rand_choice > 0.5)

        # 最终丢弃 lip 的 Batch 索引掩码
        drop_lip_mask = do_drop_mask & (rand_choice <= 0.5)

        # --- 4. 应用置零操作 (使用掩码索引对 (F, T) 块进行操作) ---
        
        # 注意：使用索引赋值操作 features[mask] = 0.0，PyTorch 会自动将 0.0 广播到 (F, T) 形状。
        features_video_face[drop_face_mask] = 0.0
        features_video_lip[drop_lip_mask] = 0.0
        
        return features_video_face, features_video_lip

    def forward(
        self,
        source: torch.Tensor,
        # target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']

        src_video_face, src_video_lip = src_video
        if mask:
            if self.masking_type == 'input':
                # target_list not used here.
                src_video_face, src_video_lip = self.apply_input_mask(src_video, padding_mask)
                # src_video_lip = self.apply_input_mask(src_video_lip, padding_mask)
                src_audio = self.apply_input_mask(src_audio, padding_mask)
            elif self.masking_type == "input2":
                src_video_face, src_video_lip = self.apply_input_mask_hubert(src_video, padding_mask)
                src_audio = self.apply_input_mask_hubert(src_audio, padding_mask)
            elif self.masking_type == 'feature':
                pass
            else:
                assert False
            

        features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
        features_video_face = self.forward_features(src_video_face, modality = 'video_face')
        features_video_lip = self.forward_features(src_video_lip,  modality = "video_lip")
        face_mask, lip_mask = source["face_lip_mask"]

        features_video_face = features_video_face * face_mask.unsqueeze(1).unsqueeze(1).to(features_video_face.dtype)
        features_video_lip = features_video_lip * lip_mask.unsqueeze(1).unsqueeze(1).to(features_video_lip.dtype)


        features_video_face, features_video_lip = self.enhanced_feature_drop_bft(features_video_face, features_video_lip, face_mask, lip_mask)        
        
        if self.masking_type == 'feature' and mask:
            assert False, "FIX ME!"
            features_audio, mask_indices_audio = self.apply_feature_mask(features_audio.transpose(1,2).clone(), padding_mask, "audio")
            features_video, mask_indices_video = self.apply_feature_mask(features_video.transpose(1,2).clone(), padding_mask, "video")
            features_audio = features_audio.transpose(1,2)
            features_video = features_video.transpose(1,2)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()

        only_video = False
        only_audio = False

        if self.modality_scheduler:
            if modality_drop_prob < self.modality_dropout:
                features_audio = 0 * features_audio
                only_video = True
        else:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                    only_video = True
                else:
                    features_video_face = 0 * features_video_face
                    features_video_lip = 0 * features_video_lip
                    only_audio = True
                
        
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video_face, features_video_lip], dim=1)
        
        elif self.modality_fuse == 'add':
            features = features_audio + features_video_face + features_video_lip
        

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2) # [b, c, t] -> [b, t, c]
        features = self.layer_norm(features)

        if padding_mask is not None:
            # subsample the padding mask
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        if self.encoder_type== "qwen" or self.encoder_type == 'qwenconv':
            x = self.encoder(
                features,
                padding_mask=padding_mask,
            )
            x = x[0]
        else:
            x, _ = self.encoder(
                features,
                padding_mask=padding_mask,
                layer=None if output_layer is None else output_layer - 1
            )


        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        cls_x = self.linear(x) 
        cls_x = cls_x.reshape(-1, cls_x.shape[-1] // 2) #[B, T, D] -> [B*T, D]


        # 2) 构造一维的保留 mask（True = 保留；False = 丢弃）
        B, T = padding_mask.shape
        asr_mask_2T = source["asr_fea_padding_mask"]   # [B, 2T]，True 表示 padding
        # 把 [B, T] 上采样到 [B, 2T]
        pad_2T = padding_mask.repeat_interleave(2, dim=1)
        # 并集：任意来源是 padding 即视为 padding
        mask_2T = pad_2T | asr_mask_2T                 # [B, 2T], True=padding
        # 降到帧级：[B, T]，一帧的两个子帧只要有一个是 padding 就算 padding
        frame_pad = mask_2T.view(B, T, 2).any(dim=2)   # [B, T], True=padding
        # 取反得到保留位置，再上采样回 [B, 2T] 并展平成 [B*2T]
        valid_mask = (~frame_pad).repeat_interleave(2, dim=1).reshape(-1)  # True=keep

        m_cls = cls_x[valid_mask]

        with torch.no_grad():
        
            fea_target = source["asr_fea"].transpose(1, 2) # b, d, t -> b, t, d
            b, t, d = fea_target.shape

            fea_target = (fea_target - self.whisper_stat[0]) / self.whisper_stat[1]

            targets = self.Quantizer(fea_target) # b, t

            m_tar = targets.reshape(-1)
            m_tar = m_tar[valid_mask]


        loss_cls = F.cross_entropy(m_cls, m_tar, reduction="sum")


        cb_vocab = m_cls.shape[-1]
        assert m_tar.max().item() <  cb_vocab 
        counts = torch.bincount(m_tar, minlength=cb_vocab)
        freq = counts.float() / counts.sum()

        predicted_classes = torch.argmax(m_cls, dim=-1)
        correct_predictions = predicted_classes == m_tar
        #pdb.set_trace()
        accuracy = (correct_predictions.sum().item() / len(correct_predictions))

      
        epsilon = 1e-12  
        H = -torch.sum(freq * (freq+epsilon).log())
        H_max = torch.log(torch.tensor(len(freq), dtype=torch.float))
        H_rel = H / H_max

        results = {
                'accuracy': accuracy,
                "sample_size":  valid_mask.long().sum(),
                "modality_dropout": self.modality_dropout,
                "codebook_maxfreq": freq.max(),
                "codebook_minfreq": freq.min(),
                "codebook_entropy": H_rel
        }

        results["losses"] = {"loss_cls": loss_cls, 
                             'loss_extra': features_pen * results["sample_size"]}
        
        for modality in ["audio", 'video', 'audio_video']:
            if only_video:
                results["video_cls"] = loss_cls.item() / results["sample_size"] 
            elif only_audio:
                results["audio_cls"] = loss_cls.item() / results["sample_size"]  
            elif (not only_video) and (not only_audio):
                results["audio_video_cls"] =  loss_cls.item() / results["sample_size"] 
            else:
                assert False

        return results

    
    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            assert False, "FIX ME"
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video_face = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
            features_video_lip = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))

        elif src_audio is None and src_video is not None:


            src_video_face, src_video_lip = src_video

            features_video_face = self.forward_features(src_video_face, modality='video_face')
            features_video_lip = self.forward_features(src_video_lip, modality='video_lip')
            features_audio = features_video_face.new_zeros(features_video_face.size(0), self.encoder_embed_dim, features_video_face.size(-1))
        
        elif src_audio is not None and src_video is not None:
            # print('audio-video')
            src_video_face, src_video_lip = src_video
            features_video_face = self.forward_features(src_video_face, modality='video_face')
            features_video_lip = self.forward_features(src_video_lip, modality='video_lip')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        face_mask, lip_mask = source["face_lip_mask"]

        features_video_face = features_video_face * face_mask.unsqueeze(1).unsqueeze(1).to(features_video_face.dtype) 
        features_video_lip = features_video_lip * lip_mask.unsqueeze(1).unsqueeze(1).to(features_video_lip.dtype)

        # 训练随机丢弃 face or lip，测试不丢弃能用都用
        if self.training:
            features_video_face, features_video_lip = self.enhanced_feature_drop_bft(features_video_face, features_video_lip, face_mask, lip_mask)
        
        # else:
        #     print('Testing')

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video_face, features_video_lip], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video_face + features_video_lip
        
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        # x, _ = self.encoder(
        #     x,
        #     padding_mask=padding_mask,
        #     layer=None if output_layer is None else output_layer - 1
        # )

        if self.encoder_type== "qwen" or self.encoder_type == 'qwenconv':
            # for visualization only 
            if False:
                assert(len(features) == 1)
                x = self.encoder(
                    features,
                    padding_mask=padding_mask,
                    output_hidden_states = True
                )
                x, all_hiddens = x[0], x[1]

                all_hiddens = torch.stack(all_hiddens).squeeze()
                print(all_hiddens.shape)
                all_hiddens = all_hiddens.detach().cpu().numpy()
                print(all_hiddens.dtype)

                filename = source['utt_id'][0]
                filename = filename.split('/')[-2:]
                filename = '/'.join(filename)
                filename = filename.replace('.wav', '.npy')
                root_dir = "/work2/asrprg/anonymous/lippretrain/force_align/base_flare/face_lip_audio_video_pt"

                filename = os.path.join(root_dir, filename)

                print(filename)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.save(filename, all_hiddens)
            
            else:
                x = self.encoder(
                    features,
                    padding_mask=padding_mask,
                )
                x = x[0]
        else:
            x, _ = self.encoder(
                features,
                padding_mask=padding_mask,
                layer=None if output_layer is None else output_layer - 1
            )


        return x, padding_mask


    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.regression_proj = None
        self.whisper = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits



    
