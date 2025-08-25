# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.ModuleList([
            nn.Linear(llm_output_size, speech_token_size + 1) for _ in range(5)
        ])
        
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_targets = [[torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i+count, :speech_token_len[i]+count].tolist() +
                                        [self.speech_token_size] + [IGNORE_ID] * (count)) for i in range(text_token.size(0))] for count in range(5)]

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = [self.llm_decoder[i](lm_output) for i in range(5)]

        # 定义损失和准确率列表
        losses = [
            [self.criterion_ce(logits[i], lm_targets[i]) for i in range(5)]
        ]

        accs = [
            th_accuracy(logits[i].view(-1, self.speech_token_size + 1), lm_targets[i], ignore_label=IGNORE_ID) for i in range(5)
        ]

        # 使用 torch.stack 和 torch.sum/mean 计算总损失和准确率
        total_loss = torch.stack(losses).sum()  # 或者使用 .mean() 计算平均值
        total_acc = torch.stack(accs).mean()    # 通常准确率用平均值更合理

        return {'loss': total_loss, 'acc': total_acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        if self.fp16 is True:
            embedding = embedding.half()
        elif self.bf16 is True:
            embedding = embedding.bfloat16()
            
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        masks = masks.to(xs.device).to(xs.dtype)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache
    
class Qwen2LM(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
            head_num: int = 5,
            inference_head_num: int = 3,
            mtp_head_num: int = 14,
            freeze_embedding: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm

        self.mtp_block = nn.ModuleList([
            Qwen2DecoderLayer(Qwen2Config(hidden_size=llm_input_size, num_attention_heads=mtp_head_num, num_key_value_heads=mtp_head_num), 0) for _ in range(head_num)
        ])

        self.llm_decoder = nn.ModuleList([
            nn.Linear(llm_output_size, speech_token_size + 3) for _ in range(head_num)
        ])

        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio
        self.head_num = head_num
        self.inference_head_num = inference_head_num

        if freeze_embedding:
            self.speech_embedding.weight.requires_grad = False
            for module in self.llm_decoder:
                module.weight.requires_grad = False
            self.llm.model.model.embed_tokens.weight.requires_grad = False

    def pad_unpad_sequence(self, sos_eos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    @staticmethod
    def pad_tensor(tensor, padding_tensor):
        """Pad a list of 2D tensors (Li, D) into a (B, Lmax, D) tensor.

        Args:
            tensor: list[torch.Tensor], each with shape (Li, D) and same D.
            padding_tensor: torch.Tensor used as row pad vector. Accepts
                shapes (D,), (1, D) or (1, 1, D). Device/dtype define output.

        Returns:
            torch.Tensor with shape (B, Lmax, D), padded with padding_tensor rows.
        """
        # Handle empty input
        if tensor is None or len(tensor) == 0:
            pad_vec = padding_tensor
            if pad_vec.dim() == 3:
                D = pad_vec.size(-1)
            elif pad_vec.dim() == 2:
                D = pad_vec.size(-1)
            else:
                D = pad_vec.numel()
            return torch.zeros(0, 0, D, device=pad_vec.device, dtype=pad_vec.dtype)

        # Normalize pad vector to shape (D,)
        pad_vec = padding_tensor
        if pad_vec.dim() == 3:
            pad_vec = pad_vec.squeeze(0).squeeze(0)
        elif pad_vec.dim() == 2:
            pad_vec = pad_vec.squeeze(0)
        # Now pad_vec should be (D,)

        # Infer sizes
        max_len = max(x.size(0) for x in tensor)
        D = tensor[0].size(1)
        B = len(tensor)

        if pad_vec.numel() != D:
            raise ValueError(f"padding_tensor last dim {pad_vec.numel()} != feature dim {D}")

        # Pre-fill with pad rows
        out = pad_vec.view(1, 1, D).expand(B, max_len, D).clone()

        # Copy each sequence to the front part
        for i, x in enumerate(tensor):
            L = x.size(0)
            if L > 0:
                out[i, :L, :] = x

        return out

    def pad_unpad_sequence_multi_head(self, sos_eos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        batch_size = text_token.size(0)

        text_token_list = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token_list = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)

        lm_input_list = []
        for i in range(batch_size):
            # 1. encode text_token
            current_batch_text_token_embedding = self.llm.model.model.embed_tokens(text_token_list[i])
            current_batch_speech_token_embedding = self.speech_embedding(speech_token_list[i])

            lm_input_list.append(torch.concat([sos_eos_emb.squeeze(dim=0), current_batch_text_token_embedding, task_id_emb.squeeze(dim=0), current_batch_speech_token_embedding], dim=0))
        
        pad_vec = self.speech_embedding.weight[self.speech_token_size].to(text_token.device)
        
        input_len = torch.tensor([i.size(0) for i in lm_input_list], dtype=torch.int32)

        lm_input_tensor = self.pad_tensor(lm_input_list, pad_vec)

        lm_target_per_head_batch = [
                    [ torch.tensor([IGNORE_ID] * (1 + text_token_len[i]) + speech_token_list[i][count:speech_token_len[i]+count].tolist() +
                                [self.speech_token_size] + [IGNORE_ID] * count) for i in range(batch_size)]
                for count in range(self.head_num)]
        
        for head in range(self.head_num):
            lm_target_per_head_batch[head] = pad_sequence(lm_target_per_head_batch[head], batch_first=True, padding_value=IGNORE_ID).to(text_token.device)

        return lm_input_tensor.to(text_token.device), lm_target_per_head_batch, input_len

    def forward(
            self,
            **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = kwargs.get('text_token')
        if not isinstance(text_token, torch.Tensor):
            raise ValueError("text_token must be a torch.Tensor")
        
        text_token_len = kwargs.get('text_token_len')
        if not isinstance(text_token_len, torch.Tensor):
            raise ValueError("text_token_len must be a torch.Tensor")
        
        speech_token = kwargs.get('speech_token')
        if not isinstance(speech_token, torch.Tensor):
            raise ValueError("speech_token must be a torch.Tensor")
        
        speech_token_len = kwargs.get('speech_token_len')
        if not isinstance(speech_token_len, torch.Tensor):
            raise ValueError("speech_token_len must be a torch.Tensor")

        # 1. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_input_tensor, lm_target_per_head_batch, input_len = self.pad_unpad_sequence_multi_head(sos_eos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len)


        # 验证lm_input和lm_targets的维度是否一致
        for i in range(self.head_num):
            assert lm_input_tensor.shape[1] == lm_target_per_head_batch[i].shape[1], f"lm_input和lm_targets的维度不一致: {lm_input_tensor.shape[1]} != {lm_target_per_head_batch[i].shape[1]}"

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input_tensor, input_len)
        mtp_output = [self.mtp_block[i](lm_output.transpose(0, 1))[0].transpose(0, 1) for i in range(self.head_num)]
        logits = [self.llm_decoder[i](mtp_output[i]) for i in range(self.head_num)]

        for i in range(self.head_num):
            assert logits[i].shape[:2] == lm_target_per_head_batch[i].shape, \
                f"expected logits (B,L,C) to match target (B,L), got {logits[i].shape} vs {lm_target_per_head_batch[i].shape}"

        # 定义损失和准确率列表
        losses = [
            self.criterion_ce(logits[i], lm_target_per_head_batch[i]) for i in range(self.head_num)
        ]
        accs = [
            th_accuracy(logits[i].view(-1, self.speech_token_size + 3), lm_target_per_head_batch[i], ignore_label=IGNORE_ID) for i in range(self.head_num)
        ]
        # 使用 torch.stack 和 torch.sum/mean 计算总损失和准确率
        total_loss = torch.stack(losses).mean()
        total_acc = torch.stack(accs).mean()

        return {'loss': total_loss, 'acc': total_acc}


    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text_local = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text_local = self.llm.model.model.embed_tokens(text_local)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_local.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, text_local, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, _ = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=None)
            # Get predictions from all heads
            mtp_output = [self.mtp_block[j](y_pred[:,-1,:].unsqueeze(1))[0] for j in range(self.inference_head_num)]
            logps = [self.llm_decoder[j](mtp_output[j][:, -1]).log_softmax(dim=-1) for j in range(self.inference_head_num)]
            # Get top ids from each head
            top_ids = [self.sampling_ids(logps[p].squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i * self.inference_head_num < min_len else False).item() for p in range(self.inference_head_num)]

            # Collect valid top ids (not EOS and not exceeding speech_token_size)
            valid_top_ids = []
            for head_id in top_ids:
                if head_id == self.speech_token_size:
                    # If any head predicts EOS, we break
                    break
                if head_id > self.speech_token_size:
                    continue
                valid_top_ids.append(head_id)
                # Yield tokens for output
                yield head_id
                out_tokens.append(head_id)
            # Check if we reached EOS
            if self.speech_token_size in valid_top_ids:
                break
            
            # Create a concatenated input of all predicted token embeddings
            if valid_top_ids:
                # Convert valid token ids to embeddings and concatenate them
                token_embeddings = [self.speech_embedding.weight[tid].reshape(1, 1, -1) for tid in valid_top_ids]
                token_embeddings = torch.cat(token_embeddings, dim=1)
                lm_input = torch.cat([lm_input, token_embeddings], dim=1)
            else:
                break