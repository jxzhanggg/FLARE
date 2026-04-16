# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile
import pickle

DBG=True if len(sys.argv) == 1 else False

if DBG:
    import utils as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
else:
    from . import utils as custom_utils

logger = logging.getLogger(__name__)

def parse_path(path):
    return path.replace('/yrfs4/', '/train8/').replace("anonymous2", 'anonymous')

                        
NUM=3000000000000000000000000000000

def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    def is_audio_label_aligned(audio_dur, label_durs):
        return all([abs(audio_dur - label_dur)<tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []

    
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2]) # 
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                face_video_path = items[1]
                lip_video_path = items[2]

                if face_video_path == 'null':
                    face_video_path = None
                
                if lip_video_path == 'null':
                    lip_video_path = None
                
                audio_path = items[3]
                audio_id = items[0]
                names.append(((face_video_path, lip_video_path), audio_path+':'+audio_id))
                inds.append(ind)
                sizes.append(sz)
            if ind >= NUM:
                break

    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):

    with open(label_path) as f:
        # f.seek 是对应的字节(Byte)数
        # f.read 是对应的字符数
        code_lengths = [[len(line.encode('utf-8')), len(line)] for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"

        starts = [byte for byte, nc in code_lengths]
        starts = list(itertools.accumulate([0] + starts))[:-1]
        
        offsets = [(start, n) for start, (byte, n) in zip(starts, code_lengths)]
        offsets = [offsets[i] for i in inds]
    
    return offsets



def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for i, line in enumerate(f) if i <= NUM]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class AVHubertDataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            load_mel = False,
            avg = 1,
            load_whisper_fea = True,
    ):
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(manifest_path, max_keep_sample_size, min_keep_sample_size, frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"

        if self.is_s2s:
            if store_labels:
                self.label_list = [load_label(p, inds, tot) for p in label_paths]
            else:
                self.label_paths = label_paths
                self.label_offsets_list = [
                    load_label_offset(p, inds, tot) for p in label_paths
                ]
            assert (
                label_processors is None
                or len(label_processors) == self.num_labels)
        
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.training = True
            self.transform = custom_utils.Compose([
                custom_utils.Resize((96, 96)),
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ])
        else:
            self.training = False
            self.transform = custom_utils.Compose([
                custom_utils.Resize((96, 96)),
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

        self.testing = False
        self.load_mel = load_mel
        self.avg = avg
        if self.load_mel:
            N_MELS = 80
            # whisper mel filter bank params
            with np.load("mel_filters.npz") as f:
                filters = torch.from_numpy(f[f"mel_{N_MELS}"])
            self.filters = filters
        self.load_whisper_fea = load_whisper_fea
    

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, n = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(n)
                label = label.strip()

        if self.label_processors is not None:
            def _process_sil(text, processor):

                text_no_sils = text.split('<sil>')
                labels = []
                unk = torch.tensor(3)
                for i, t in enumerate(text_no_sils):
                    label = processor(t)
                    if i < len(text_no_sils) -1:
                        label = label[:-1] # remove eos
                        labels.extend(label)
                        labels.append(unk)
                    else:
                        labels.extend(label)
                
                return torch.tensor(labels)
            
            label = _process_sil(label, self.label_processors[label_idx])

        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]
    

    def convert_fb80_whisper(self, np_data, N_FFT=400, N_MELS=80, HOP_LENGTH=160):
        data = torch.from_numpy(np_data)
        window = torch.hann_window(N_FFT) #         #.to(data.device)
        stft = torch.stft(data, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[:, :-1].abs() ** 2
        # self.filters = self.filters.to(data.device)
        mel_spec = torch.matmul(self.filters, magnitudes)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec.transpose(0, 1)
        return log_spec.numpy()
    

    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, audio_fn = mix_name

        
        if 'audio' in self.modalities:
            audio_fn = audio_fn.split(':')[0]
            sample_rate, wav_data = wavfile.read(audio_fn)
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            
            if self.load_mel:
                clean_mel = self.convert_fb80_whisper(wav_data.astype(np.float32) / 32768.0)
            else:
                clean_mel = None

            if np.random.rand() < self.noise_prob:
                noise_wav_data = self.add_noise(wav_data)
            else:
                noise_wav_data = wav_data
            
            audio_feats = logfbank(noise_wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
            audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
            
        else:
            audio_feats = None
            audio_fn = audio_fn.split(':')[0]
            clean_mel = None


        if 'video' in self.modalities:
            
            
            face_video_fn, lip_video_fn = video_fn
            face_video_feats, lip_video_feats = None, None
            if face_video_fn is not None:
                face_video_feats = self.load_video(face_video_fn) # [T, H, W, 1]
            if lip_video_fn is not None:
                lip_video_feats = self.load_video(lip_video_fn)

            # assert(not ((face_video_feats is None) and (lip_video_feats is None)))

            if (face_video_feats is not None) and  (lip_video_feats is not None):
                video_len = min(len(face_video_feats), len(lip_video_feats))

                face_video_feats = face_video_feats[:video_len]
                lip_video_feats = lip_video_feats[:video_len]

            elif face_video_feats is not None:
                video_len = len(face_video_feats)
            elif lip_video_feats is not None:
                video_len = len(lip_video_feats)
            
            else:
                pass
            
            # video_feats = np.random.normal(size=(t, 88, 88, 1))
            video_feats = (face_video_feats, lip_video_feats)

            if (face_video_feats is None) and (lip_video_feats is None):
                video_feats = None

        else:
            video_feats = None

        if not self.testing and self.load_whisper_fea:  
            # load offline whisper deep features.    
            if self.avg == 1:         
                wavlm_fea_path = audio_fn.replace("/audio/", '/whisper_fea/layer_last1/').replace('.wav', '.npy')
            elif self.avg == 8:
                wavlm_fea_path = audio_fn.replace("/audio/", '/whisper_fea/layer_last8_avg8/').replace('.wav', '.npy')
            asr_fea = np.load(wavlm_fea_path) # t, d
            #asr_fea = np.zeros((t*2, 1280))
        else:
            asr_fea = None

        
        # 以video长度为基准，左对齐
        if (audio_feats is not None) and (video_feats is not None):
            
            
            diff = len(audio_feats) - video_len
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        
        return video_feats, audio_feats, asr_fea, clean_mel

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        if feats is None:
            return None
        
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):

        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:

            noise_wav.append(wavfile.read(parse_path(self.noise_wav[x]))[1].astype(np.float32))
        
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed


    def __getitem__(self, index):
        video_feats, audio_feats, asr_fea, clean_mel = self.load_feature(self.names[index])
        audio_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None

        if video_feats is None:
            video_feats = None
        else:
            face_video_feats, lip_video_feats  = video_feats
            face_video_feats = torch.from_numpy(face_video_feats.astype(np.float32)) if face_video_feats is not None else None
            lip_video_feats = torch.from_numpy(lip_video_feats.astype(np.float32)) if lip_video_feats is not None else None
            video_feats = (face_video_feats, lip_video_feats)


        asr_fea = torch.from_numpy(asr_fea.astype(np.float32)) if asr_fea is not None else None
        clean_mel = torch.from_numpy(clean_mel.astype(np.float32)) if clean_mel is not None else None

        if video_feats is None and ("video" in self.modalities):
            fid = self.names[index][1].split(':')[0]
            logger.info(f'WARNING: index {index} video {fid} is None.')
            return {"id": None}
        
        assert(self.normalize)
        if self.normalize and (audio_feats is not None):
            with torch.no_grad():
                # T, D; norm over D
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        if self.is_s2s:
            labels = self.get_labels(index)
        else:
            labels = None

        fid = self.names[index][1].split(':')[0]
        return {"id": index, 'fid': fid, "video_source": video_feats, 'audio_source': audio_feats, "label_list": labels, "asr_fea": asr_fea, "clean_mel": clean_mel}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        

        asr_fea = [s["asr_fea"] for s in samples]

        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(face) if face is not None else len(lip) for face, lip in video_source]
        
        assert(self.pad_audio==True)
        
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)

        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        
        if video_source is not None:
            # collated_videos = video_source
            collated_videos, face_lip_mask,  padding_mask, audio_starts = self.collater_video(video_source, audio_size, audio_starts)
        else:
            collated_videos = None
        
        # [b, f, t]
        assert(np.all(np.array(audio_starts)==0)), f"audio_starts {audio_starts}"

        if (not self.testing) and self.load_whisper_fea:
            collated_asr_fea, asr_fea_padding_mask, _ = self.collater_audio(
                asr_fea,
                audio_size * 2,
                None)
        else:
            collated_asr_fea, asr_fea_padding_mask = None, None
        
        clean_mel = [s["clean_mel"] for s in samples]
        if clean_mel[0] is not None:
            collated_clean_mel, _, _ = self.collater_audio(clean_mel, 3000)
        else:
            collated_clean_mel = None


        if self.is_s2s:
            targets_by_label = [
                [s["label_list"][i] for s in samples]
                for i in range(self.num_labels)
            ]
            targets_list, lengths_list, ntokens_list = self.collater_label(
                targets_by_label, audio_size, audio_starts
            )
        
        source = {"audio": collated_audios, 
                "video": collated_videos, 
                "face_lip_mask": face_lip_mask,
                "asr_fea": collated_asr_fea,
                "asr_fea_padding_mask":asr_fea_padding_mask,  
                "clean_mel": collated_clean_mel, 
                "utt_id": [s['fid'] for s in samples]}
        
        net_input = {"source": source, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples]
        }


        if self.single_target:
            if self.is_s2s:
                batch["target_lengths"] = lengths_list[0]
                batch["ntokens"] = ntokens_list[0]
                batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
                batch["net_input"]["target"] = targets_list[0][0]
                batch["net_input"]["target_lengths"] = lengths_list[0]

        return batch

    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False) # 
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts
    

    def collater_video(self, videos, audio_size, audio_starts=None):
    
        # audio_feat_shape = list(audios[0].shape[1:])
        # video_feat_shape = [88, 88, 1]

        face, lip = videos[0]

        dummy_tensor = face if face is not None else lip


        collated_faces = dummy_tensor.new_zeros([len(videos), audio_size, 88, 88, 1])
        collated_lips = dummy_tensor.new_zeros([len(videos), audio_size, 88, 88, 1])


        padding_mask =  torch.BoolTensor(len(videos), audio_size).fill_(False) # 

        face_mask = torch.BoolTensor(len(videos)).fill_(False)
        lip_mask = torch.BoolTensor(len(videos)).fill_(False)

        start_known = audio_starts is not None


        audio_starts = [0 for _ in videos] if not start_known else audio_starts

        for i, (face, lip) in enumerate(videos):
        
            video_len = len(face) if face is not None else len(lip)

            diff = video_len - audio_size

            if face is not None:
                face_mask[i] = True
            
            if lip is not None:
                lip_mask[i] = True
            

            if diff == 0:
                
                if face is not None:
                    collated_faces[i] = face
                
                if lip is not None:
                    collated_lips[i] = lip
            
            elif diff < 0:
                assert self.pad_audio

                if face is not None:
                    collated_faces[i] = torch.cat(
                        [face, face.new_full([-diff, 88, 88, 1], 0.0)]
                    )
                if lip is not None:
                    collated_lips[i] = torch.cat(
                        [lip, lip.new_full([-diff, 88, 88, 1], 0.0)]
                    )
                
                padding_mask[i, diff:] = True
            else:
                
                if face is not None:
                    collated_faces[i], audio_starts[i] = self.crop_to_max_size(
                        face, audio_size, audio_starts[i] if start_known else None
                    )
                if lip is not None:
                    collated_lips[i], audio_starts[i] = self.crop_to_max_size(
                        lip, audio_size, audio_starts[i] if start_known else None
                    )
            

        collated_faces = collated_faces.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        collated_lips = collated_lips.permute((0, 4, 1, 2, 3)).contiguous()

        return (collated_faces, collated_lips), (face_mask, lip_mask), padding_mask, audio_starts


    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        # sample rate: how many  a/v frame per second
        # label rate: how many label per second
        s2f = label_rate / self.sample_rate # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label_s2s(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = self.label_processors[0].dictionary.pad(), self.label_processors[0].dictionary.eos()
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
        prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
        return (targets_, prev_output_tokens), lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                if self.is_s2s:
                    targets, lengths, ntokens = self.collater_seq_label_s2s(targets, pad)
                else:
                    targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]
