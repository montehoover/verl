import asyncio
import os
import time
import json
import uuid

import datasets
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


from verl.compliance.constants import COT_OPENING, COT_OPENING_QWEN, LABEL_OPENING, NEG_LABEL, POS_LABEL

import logging
logger = logging.getLogger(__name__)


class ComplianceProjectError(ValueError):
    pass

class ModelWrapper:
    def get_message_template(self, system_content=None, user_content=None, assistant_content=None):
        # assistant_content = assistant_content or LABEL_OPENING
        message = []
        if system_content is not None:
            message.append({'role': 'system', 'content': system_content})
        if user_content is not None:
            message.append({'role': 'user', 'content': user_content})
        if assistant_content is not None:
            message.append({'role': 'assistant', 'content': assistant_content})
        if not message:
            raise ComplianceProjectError("No content provided for any role.")
        return message

    def get_message_template_cot(self, system_content, user_content, assistant_content=None):
        assistant_content = assistant_content or COT_OPENING
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ]

    def apply_chat_template(self, system_content=None, user_content=None, assistant_content=None):
        return self.get_message_template(system_content, user_content, assistant_content)
    
    # def apply_chat_template_cot(self, system_content, user_content, assistant_content=None):
    #     return self.get_message_template_cot(system_content, user_content, assistant_content)


class LocalModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000):
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens
        if "nemoguard" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
    
    def apply_chat_template(self, system_content, user_content, assistant_content=None, enable_thinking=True):
        if assistant_content is not None:
            # This works for both Qwen3 and non-Qwen3 models, and any time assistant_content is provided, it automatically adds the <think></think> pair before the content like we want for Qwen3 models.
            message = self.get_message_template(system_content, user_content, assistant_content)
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        elif enable_thinking:
            if "qwen3" in self.model_name.lower():
                # Let the Qwen chat template handle the thinking token
                message = self.get_message_template(system_content, user_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                # The way the Qwen3 chat template works is it adds a <think></think> pair when enable_thinking=False, but for enable_thinking=True, it adds nothing. We want to force the token to be there.
                prompt = prompt + f"\n{COT_OPENING_QWEN}"
            else:
                message = self.get_message_template(system_content, user_content, assistant_content=COT_OPENING_QWEN)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        else:
            # This works for both Qwen3 and non-Qwen3 models. 
            # When Qwen3 gets assistant_content, it automatically adds the <think></think> pair before the content like we want. And other models ignore the enable_thinking argument.
            message = self.get_message_template(system_content, user_content, assistant_content=f"{LABEL_OPENING}\n")
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True, enable_thinking=False)
        return prompt


class HfModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000):
        super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
    
    def get_prediction(self, message, strict=False):
        pos_token_id = self.tokenizer.encode(POS_LABEL, add_special_tokens=False)[0]
        neg_token_id = self.tokenizer.encode(NEG_LABEL, add_special_tokens=False)[0]
        
        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits
        prediction_logits = logits[0, -1, :] # Next token prediction logits are last in the sequence
        predicted_token_id = torch.argmax(prediction_logits).item()
        if not predicted_token_id in [pos_token_id, neg_token_id] and strict:
            predicted_token = self.tokenizer.decode(predicted_token_id, skip_special_tokens=True)
            raise ComplianceProjectError(f"The next token prediction was neither {POS_LABEL} nor {NEG_LABEL}. Instead it was '{predicted_token}'. Consider debugging by getting the full generation to see what is happening.")
        
        prediction_probs = torch.nn.functional.softmax(prediction_logits, dim=-1)
        pos_prob = prediction_probs[pos_token_id].item()
        pos_logit = prediction_logits[pos_token_id].item()
        neg_logit = prediction_logits[neg_token_id].item()
        return pos_prob, (pos_logit, neg_logit)
        
    def get_prediction_probs(self, messages, strict=False):
        pos_token_probs_logits = [self.get_prediction(message, strict) for message in tqdm(messages)]
        pos_token_probs, logit_pairs = zip(*pos_token_probs_logits)
        return pos_token_probs, logit_pairs
        
    def get_response(self, message, temperature=None, top_k=None, top_p=None, logit_bias_dict=None):
        if logit_bias_dict is not None:
            token = list(logit_bias_dict.keys())[0]
            bias = float(logit_bias_dict[token])
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            logit_bias_dict = {tuple(token_ids): bias}

        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_content = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                temperature=temperature or self.temperature,
                top_k=top_k or self.top_k,
                top_p=top_p or self.top_p,
                min_p=self.min_p,
                pad_token_id=self.tokenizer.pad_token_id,
                sequence_bias=logit_bias_dict,
                renormalize_logits=True, 
            )
        # keep only the newly generated tokens
        prompt_len = inputs.input_ids.shape[-1]
        new_token_ids = output_content[:, prompt_len:]
        output_text = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        return output_text


    def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None):
        outputs = [self.get_response(message, temperature, top_k, top_p, logit_bias_dict) for message in tqdm(messages)]
        return outputs


class VllmModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, max_model_len=8192):
        from vllm import LLM, SamplingParams
        super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens)
        self.model = LLM(
            model=model_name, 
            max_model_len=max_model_len, 
            gpu_memory_utilization=0.95,
        )

    def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None):
        if logit_bias_dict is not None:
            token = list(logit_bias_dict.keys())[0]
            bias = float(logit_bias_dict[token])
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            logit_bias_dict = {token_id: bias for token_id in token_ids}

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature or self.temperature,
            top_k=top_k or self.top_k,
            top_p=top_p or self.top_p,
            min_p=self.min_p,
            seed=time.time_ns(),
            logit_bias=logit_bias_dict,
        )
        # responses -> List[obj(prompt, outputs -> List[obj(text, ???)])]
        responses = self.model.generate(messages, sampling_params=sampling_params)
        outputs = [response.outputs[0].text for response in responses]
        return outputs
    
    def get_prediction_probs(self, messages, strict=False):
        pos_token_id = self.tokenizer.encode(POS_LABEL, add_special_tokens=False)[0]
        neg_token_id = self.tokenizer.encode(NEG_LABEL, add_special_tokens=False)[0]
    
        sampling_params = SamplingParams(max_tokens=1, logprobs=20)
        # responses -> List[obj(prompt, outputs -> List[obj(text, logprobs, index, token_ids, cumulative_logprobs)])]
        # loggprobs -> List[{token_id: obj(logprob, rank, decoded_token)}]
        responses = self.model.generate(messages, sampling_params=sampling_params)
        pos_token_probs = []
        for response in responses:
            token_logprob_dict = response.outputs[0].logprobs[0]
            if not (pos_token_id in token_logprob_dict or neg_token_id in token_logprob_dict) and strict:
                raise ComplianceProjectError(f"The next token prediction was neither {POS_LABEL} nor {NEG_LABEL}. Instead we got '{token_logprob_dict}'. Consider debugging by getting the full generation to see what is happening.")
            logprob_obj = token_logprob_dict.get(pos_token_id, None)
            if logprob_obj is not None:
                logprob = logprob_obj.logprob
            else:
                logprob = -100
            pos_token_prob = self._logprob_to_prob(logprob)
            pos_token_probs.append(pos_token_prob)
        logit_pairs = None
        return pos_token_probs, logit_pairs
    
    def _logprob_to_prob(self, logprob):
        min_safe_input = torch.log(torch.tensor(torch.finfo(torch.float32).tiny))
        max_safe_input = torch.log(torch.tensor(torch.finfo(torch.float32).max))
        logprob = torch.clamp(torch.tensor(logprob), min=min_safe_input, max=max_safe_input)
        prob = torch.exp(logprob).item()
        return prob
    