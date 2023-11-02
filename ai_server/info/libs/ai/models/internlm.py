# *_*coding:utf-8 *_*
# @Author : YueMengRui
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class InternLM:

    def __init__(self, model_name_or_path, model_name, logger=None, device='cuda', max_length=16 * 1024,
                 max_new_tokens=2048, **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.logger = logger
        self._load_model(model_name_or_path, device)
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        if self.logger:
            self.logger.info(str({'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens}))

        # warmup
        self.lets_chat('你好', [], stream=False)

    def _load_model(self, model_name_or_path, device):

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
            .to(torch.bfloat16)
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = self.model.device

    def get_embeddings(self, sentences):
        embeddings = []
        for text in sentences:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            model_output = self.model(input_ids, output_hidden_states=True)
            data = model_output.hidden_states[-1][0]
            data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)

            embeddings.append(data)

        return embeddings

    def check_token_len(self, prompt: str):
        code = True
        prompt_token_len = self.token_counter(f"""<|User|>:{prompt}<eoh>\n<|Bot|>:""")
        if prompt_token_len > self.max_length:
            code = False

        return code, prompt_token_len, self.max_length, self.model_name

    def token_counter(self, prompt):
        return len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])

    def select_history(self, prompt, history, max_prompt_length):
        base_prompt_token_num = self.token_counter(f"""<|User|>:{prompt}<eoh>\n<|Bot|>:""")
        true_history = []
        if history and base_prompt_token_num < max_prompt_length:
            for (old_query, old_response) in history[::-1]:
                history_token_num = self.token_counter(
                    f"""<s><|User|>:{old_query}<eoh>\n<|Bot|>:{old_response}<eoa>\n""")

                if base_prompt_token_num + history_token_num > max_prompt_length:
                    break
                else:
                    true_history.insert(0, [old_query, old_response])
                    base_prompt_token_num += history_token_num

        return true_history

    def build_prompt(self, query: str, history=[]):
        prompt = ""
        for record in history:
            prompt += f"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return prompt

    def lets_chat(self, prompt, history, stream, generation_configs={}, **kwargs):

        if not (('max_new_tokens' in generation_configs) and (
                isinstance(generation_configs['max_new_tokens'], int)) and (
                        128 < generation_configs['max_new_tokens'] < self.max_length)):
            generation_configs.update({'max_new_tokens': self.max_new_tokens})

        generation_configs.update({'max_length': self.max_length})
        max_prompt_length = self.max_length - generation_configs['max_new_tokens']

        if self.logger:
            self.logger.info(
                str({'max_prompt_length': max_prompt_length, 'generation_configs': generation_configs}) + '\n' + str(
                    kwargs) + '\n')

        history = self.select_history(prompt, history, max_prompt_length)
        input_prompt = self.build_prompt(prompt, history)
        prompt_tokens = self.token_counter(input_prompt)

        if self.logger:
            self.logger.info(str({'prompt_tokens': prompt_tokens, 'prompt_str_len': len(input_prompt),
                                  'prompt': input_prompt}) + '\n')

        if stream:
            def stream_generator():
                start = time.time()
                for resp in self.model.stream_chat(tokenizer=self.tokenizer,
                                                   query=prompt,
                                                   history=history,
                                                   **generation_configs,
                                                   **kwargs):
                    generation_tokens = self.token_counter(resp[0])
                    time_cost = time.time() - start
                    average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                    torch_gc(self.device)

                    yield {"model_name": self.model_name,
                           "answer": resp[0],
                           "history": history,
                           "time_cost": {"generation": f"{time_cost:.3f}s"},
                           "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                                     "total_tokens": prompt_tokens + generation_tokens, "average_speed": average_speed}}

            return stream_generator()
        else:
            start = time.time()
            answer, _ = self.model.chat(tokenizer=self.tokenizer,
                                        query=prompt,
                                        history=history,
                                        **generation_configs,
                                        **kwargs)
            generation_tokens = self.token_counter(answer)
            time_cost = time.time() - start
            average_speed = f"{generation_tokens / time_cost:.3f} token/s"
            torch_gc(self.device)

            return {"model_name": self.model_name,
                    "answer": answer,
                    "history": history,
                    "time_cost": {"generation": f"{time_cost:.3f}s"},
                    "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                              "total_tokens": prompt_tokens + generation_tokens, "average_speed": average_speed}}
