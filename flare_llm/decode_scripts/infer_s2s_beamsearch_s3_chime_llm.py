# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sklearn

import faulthandler
import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import numpy as np
import torch
import torch.nn as nn
from fairseq import options, tasks, utils, distributed_utils
#import sys
#print(sys.path)
import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from utils import json2wer


#print()
#config_path = Path(__file__).resolve().parent / "conf"
config_path = os.path.join(os.getcwd(), 'conf')

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
   
    )
    temp: float = 1.0


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}



def beam_search(fake_input_ids, 
                hidden_states, 
                model, 
                max_length=512, 
                real_beam_size=10, 
                vocab_size=65024, 
                EOS_TOKEN=2,
                temp=1.0):
    # 此时输出序列中只有sos token
    # seqs = k_prev_words #(beam_size, 1)
    # 初始化scores向量为0
    beam_size = real_beam_size
    final_top_k_scores = torch.zeros(beam_size).to(fake_input_ids.device)

    complete_seqs = list()
    complete_seqs_scores = list()

    attention_mask = None
    # attention_mask = model.llm.transformer.get_masks(
    #                 fake_input_ids, fake_input_ids.device)

    # position_ids = model.llm.transformer.get_position_ids(
    #                 fake_input_ids, device=fake_input_ids.device)
    position_ids = None

    use_cache = True

    past_key_values = None
    # presents = () if use_cache else None #保留 k, v
    # past_key_values = tuple([None] * len(model.llm.transformer.encoder.layers))
    # rotary_pos_emb = model.llm.transformer.rotary_pos_emb(fake_input_ids.shape[1])

    step = 1

    if model.use_lora:
        QWEN = model.llm.model.model
        QWENLM = model.llm.model
    else:
        QWEN = model.llm.model
        QWENLM = model.llm

    while True:
        if step == 1:
            outputs = QWEN(
                input_ids = None,
                attention_mask = None,
                position_ids = None,
                past_key_values = None,
                inputs_embeds = hidden_states,
                use_cache=True,
                return_dict=True)
            hidden_states = outputs.last_hidden_state
            lm_logits = QWENLM.lm_head(hidden_states)
            next_token_logits = lm_logits[:,-1,:].float() # (beam_size, vocab_size)
            next_token_logits = torch.log_softmax(next_token_logits / temp, dim=-1)  
            # 因为最开始解码的时候只有一个结点<sos>,所以只需要取其中一个结点计算topk
            top_k_scores, top_k_words = next_token_logits[0].topk(beam_size, dim=0, largest=True, sorted=True)
            fake_input_ids = fake_input_ids.repeat(beam_size, 1) # beam_size, 1, h
                
        else:
            outputs = QWEN(**model_inputs,
                            return_dict=True)

            # 此时要先展开再计算topk
            # lm_logits = outputs.logits
            hidden_states = outputs.last_hidden_state
            lm_logits = QWENLM.lm_head(hidden_states)

            next_token_logits = lm_logits[:,-1,:].float() # (beam_size, vocab_size)
            next_token_logits = torch.log_softmax(next_token_logits / temp, dim=-1)
            next_token_logits = final_top_k_scores.unsqueeze(-1).expand(-1, vocab_size) + next_token_logits
            # top_k_scores: (beam_size) top_k_words: (beam_size)
            top_k_scores, top_k_words = next_token_logits.contiguous().view(-1).topk(beam_size, 0, True, True)
        beam_id = top_k_words // vocab_size  # (beam_size)  实际是beam_id
        top_k_words = (top_k_words) % vocab_size  # (beam_size)  实际是token_id
        # fake_input_ids: (beam_size, step) ==> (beam_size, step+1)
        fake_input_ids = torch.cat([fake_input_ids[beam_id], top_k_words.unsqueeze(1)], dim=1)
        # last_token_embedding = model.llm.transformer.embedding.word_embeddings(top_k_words.unsqueeze(0)) # 1, beam_size, d
        # hidden_states = torch.cat([hidden_states[:, beam_id], last_token_embedding], dim=0)

        # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
        incomplete_inds = [ind for ind, next_word in enumerate(top_k_words) if
                                next_word != EOS_TOKEN]
        # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
        complete_inds = list(set(range(len(top_k_words))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(fake_input_ids[complete_inds]) # 加入句子
            complete_seqs_scores.extend(top_k_scores[complete_inds] / step) # 加入句子对应的累加log_prob
            # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
            beam_size -= len(complete_inds) 

            if beam_size <= 0: # 完成
                break
        if step == 1:

            presents = outputs.past_key_values # .to_legacy_cache()
            new_presents = ()
            for p in presents:
                # key query -> [sk/sq, b, np, hn] # shape not right ! 
                # shape is [batch_size, num_heads, seq_len, head_dim]!!!

                # new_p = (p[0].expand(-1, real_beam_size, -1, -1)[:, incomplete_inds], p[1].expand(-1, real_beam_size, -1, -1)[:, incomplete_inds])
                new_p = (p[0].expand(real_beam_size, -1, -1, -1)[incomplete_inds], 
                        p[1].expand(real_beam_size, -1, -1, -1)[incomplete_inds])
                new_presents = new_presents + (new_p,)
            
        else:
            presents = outputs.past_key_values # .to_legacy_cache()
            new_presents = ()
            for p in presents:
                new_p = (p[0][beam_id, :, :, :][incomplete_inds], 
                        p[1][beam_id, :, :, :][incomplete_inds])
                new_presents = new_presents + (new_p,)
            
        # 更新下一次迭代数据, 仅专注于那些还没完成的句子  
        # fake_input_ids = fake_input_ids[incomplete_inds]
        # hidden_states = hidden_states[:, incomplete_inds]
            
        model_inputs = QWENLM.prepare_inputs_for_generation(fake_input_ids[incomplete_inds],
                                                    past_key_values=new_presents,
                                                    use_cache=True)

        final_top_k_scores = top_k_scores[incomplete_inds]
        fake_input_ids = fake_input_ids[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)   #(s, 1) s < beam_size

        if step > max_length: # decode太长后，直接break掉
            break
        step += 1
    
    if len(complete_seqs) == 0:
        return fake_input_ids[0], top_k_scores[0], fake_input_ids
    

    i = complete_seqs_scores.index(max(complete_seqs_scores)) # 寻找score最大的序列
    # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
    seq = complete_seqs[i] 
    score = complete_seqs_scores[i]
    return seq, score, complete_seqs, complete_seqs_scores



def search(models, sample, beam_size, audio_only, video_only, temp):

    with torch.no_grad():
        assert(len(models) == 1)
        model = models[0]

        if model.cfg.downsample:
            logger.info("Downsample is true!")
        else:
            logger.info('Downsample is false!')


        # if not audio_only:
        output_encoder = model.encoder.forward(**sample["net_input"])


        # if not video_only:
        # output_audio = model.audio_encoder(
        #     source = sample["net_input"]["source"],
        #     padding_mask = sample["net_input"]["padding_mask"],
        #     visual_features = output_video["encoder_out_video"].transpose(0, 1),
        #     visual_padding_mask = output_video["encoder_padding_mask"],
        #     tf_mask=False,
        #     adaptive_tmask=False,
        #     mask_audio_indices=None,
        #     audio_dropout = video_only,
        #     drop_video= audio_only)


        def _flatten_sents(hidden, hidden_length):
            # hidden : [T, B, D], hidden_length: [B, T]
            hidden_list = []
            for batch_idx in range(hidden.shape[1]):
                cutted_hidden = hidden[:hidden_length[batch_idx], batch_idx, :]
                hidden_list.append(cutted_hidden)
            return hidden_list
        

        # if not video_only:

        hidden_audio = output_encoder["encoder_out"] # [T, B, D]
        padding_mask = output_encoder["encoder_padding_mask"] #[B, D] True if mask

        # hidden_cat = torch.concat([hidden_audio.transpose(0, 1),
        #                             output_video["encoder_out_video"].transpose(0, 1)],
        #                             dim=-1) # B, T, D1+d2
        # hidden_fusion = model.audio_video_fusion(model.fusion_ln(hidden_cat))
        # output_fusion, _ = model.fusion_encoder(hidden_fusion,
        #                     padding_mask = padding_mask,
        #                     layer = None)

        if model.cfg.downsample:
            logger.info("Downsample is true!")
            hidden_audio = hidden_audio.permute(1, 2, 0) #t, b, d -> b, d, t
            hidden_audio = F.avg_pool1d(hidden_audio, 
                                    kernel_size=2, 
                                    stride=2, 
                                    padding=0)
            hidden_audio = hidden_audio.permute(2, 0, 1) # b, d, t -> t, b, d
            hidden_len = (~padding_mask).long().sum(dim=1)
            hidden_len = torch.ceil(hidden_len.float() / 2.).long()

        else:
            logger.info('Downsample is false!')
            hidden_len = (~padding_mask).long().sum(dim=1)        

        projected_hidden_audio = model.modality_projection(hidden_audio) #> t, b, d
        hidden_list_audio = _flatten_sents(projected_hidden_audio, hidden_len)
        hidden_list = hidden_list_audio

        logger.info(f'[prompt str] {model.prompt_str}')
        prompt_token =  model.llm_tokenizer.encode(model.prompt_str)
        bos = model.llm_tokenizer.eos_token_id
        hypos = []
        n_best_hypos = []

        device="cuda"

        if model.use_lora:
            QWEN = model.llm.model.model
        else:
            QWEN = model.llm.model
        
        A_token = model.llm_tokenizer.encode("A")
        A_embedding = QWEN.embed_tokens(torch.tensor(A_token).to(device))

        def _get_speech_embedding(batch_idx):
            projected_speech_embedding = torch.concat([A_embedding, 
                                                        hidden_list_audio[batch_idx]], dim=0)
        
            return projected_speech_embedding
        
        for batch_idx in range(len(hidden_list)):


            speech_embedding = _get_speech_embedding(batch_idx)


            gmask_sop = torch.tensor(prompt_token + [bos,]).to(speech_embedding.device)
            gmask_sop_embedding = QWEN.embed_tokens(gmask_sop) # N -> N, D
            concat_embedding = torch.concat([speech_embedding, gmask_sop_embedding], dim=0) # T, D

            fake_input_ids = torch.concat([torch.zeros(len(speech_embedding)).to(speech_embedding.device), 
                                        torch.tensor(gmask_sop)]).long()
            fake_input_ids = fake_input_ids.unsqueeze(0) # 1, N 

            concat_embedding = concat_embedding.to(torch.bfloat16)

            res, score, all_beam_res, all_beam_res_scores  = beam_search(
                fake_input_ids, 
                concat_embedding.unsqueeze(0), 
                model, 
                max_length=fake_input_ids.shape[1], 
                real_beam_size=beam_size,
                temp = temp,
                vocab_size =152064, 
                EOS_TOKEN=151643)
            
            fake_input_ids_filter = res[len(speech_embedding)+len(gmask_sop):]
            decode_string = model.llm_tokenizer.decode(fake_input_ids_filter, skip_special_tokens=True)
            hypos.append(decode_string)

            n_best = []
            for ii, b_res in enumerate(all_beam_res):
                b_hypo = model.llm_tokenizer.decode(b_res[len(speech_embedding)+len(gmask_sop):], skip_special_tokens=True)
                n_best.append((b_hypo, all_beam_res_scores[ii].item()))
            
            n_best_hypos.append(n_best)
        
        return hypos, n_best_hypos
                


def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path], strict=False)
    
    #print(task.target_dictionary)
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities

    # task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav

    # override modality!
    # not override!
    # task.cfg.modalities = cfg.override.modalities
    # I comment this, so that not override. So the audio/video inputs are all kept.
    # task.cfg.modalities = cfg.override.modalities

    audio_only = False
    video_only = False
    # to be consisitent with training !

    if "audio" in cfg.override.modalities and not "video" in cfg.override.modalities:
        models[0].get_prompt(drop_audio=False, drop_video=True, device="cuda", set_prompt=True)
        audio_only = True
    elif not "audio" in cfg.override.modalities and  "video" in cfg.override.modalities:
        models[0].get_prompt(drop_audio=True, drop_video=False, device="cuda", set_prompt=True)
        video_only = True
    elif  "audio" in cfg.override.modalities and "video" in cfg.override.modalities:
        models[0].get_prompt(drop_audio=False, drop_video=False, device="cuda", set_prompt=True)
    

    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        
        if cfg.common.bf16:
            model.to(torch.bfloat16)
        
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    

    dataset = task.dataset(cfg.dataset.gen_subset)
    dataset.testing = True
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    if cfg.generation.match_source_len:
        logger.warning(
            "The option match_source_len is not applicable to speech recognition. Ignoring it."
        )
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {
        "lm_model": lms[0], # None
        "lm_weight": cfg.generation.lm_weight, # 0.
    }

    cfg.generation.score_reference = False  #
    save_attention_plot = cfg.generation.print_alignment is not None
    cfg.generation.print_alignment = None  #
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs)

    def decode_fn(x):
        symbols_ignore = get_symbols_to_strip_from_output(generator)
        symbols_ignore.add(dictionary.pad())
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return task.datasets[cfg.dataset.gen_subset].label_processors[0].decode(x, symbols_ignore)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    result_dict_nbest = {'utt_id': [], "ref": [], 'n_best': []}

    prompt_len = 0 + 1


    def remove_sil(sent):
        
        # 
        sent = sent.replace(' <sil> ', '')
        sent = sent.replace('<sil>', '')

        return sent

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()

        hypos, n_best_hypos = search(models, sample, beam_size=cfg.generation.beam, audio_only = audio_only, video_only = video_only,
        temp = cfg.temp)

        '''

        with torch.no_grad():
            hypos =  generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )
        
        the following code equal to above code

        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        '''
        # num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)

        num_generated_tokens = sum([len(h) for h in hypos]) # batch of best string.

        gen_timer.stop(num_generated_tokens)

        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            result_dict_nbest['utt_id'].append(sample['utt_id'][i])

            ref_sent = models[0].llm_tokenizer.decode(
                sample['net_input']['target'][i].int().cpu()[prompt_len:], skip_special_tokens=True)

            result_dict_nbest["ref"].append(ref_sent)

            ref_sent = remove_sil(ref_sent)
            result_dict['ref'].append(ref_sent)

            hypo_str = hypos[i]
            hypo_str = remove_sil(hypo_str)
            result_dict['hypo'].append(hypo_str)

            n_best = n_best_hypos[i]
            result_dict_nbest["n_best"].append(n_best)

            logger.info(f"\nUTT:{result_dict['utt_id'][-1]}\nREF:{ref_sent}\nHYP:{hypo_str}\n")
        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Recognized {:,} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    result_fn_nbest = f"{cfg.common_eval.results_path}/nbest-{fid}.json"

    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    json.dump(result_dict_nbest, open(result_fn_nbest, 'w'), indent=4)

    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
    wer = 100 * n_err / n_total
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")
    logger.info(f"WER: {wer}%")
    json2wer(result_fn)
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    wer = float("inf")


    main(cfg)
    # try:
    #     if cfg.common.profile:
    #         with torch.cuda.profiler.profile():
    #             with torch.autograd.profiler.emit_nvtx():
    #                 distributed_utils.call_main(cfg, main)
    #     else:
    #         distributed_utils.call_main(cfg, main)

    # except BaseException as e:  # pylint: disable=broad-except
    #     if not cfg.common.suppress_crashes:
    #         raise
    #     else:
    #         logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
