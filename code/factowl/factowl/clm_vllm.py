# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from factowl.atomic_facts_sped_up_vllm import VLLMGenerator
from vllm import SamplingParams
from transformers import AutoTokenizer

DEFAULT_VERIFICATION_SYSTEM_PROMPT = {"en": "You are an expert fact extraction and verification assistant. " \
                                            "You are given an atomic fact and a list of textual passages. Your task is to determine whether the atomic fact " \
                                            "is True or False  based solely on the information in the passages.\n" \
                                            "Instructions:\n" \
                                            "1. Check if any of the passages directly support the atomic fact.\n" \
                                            "2. Output 'True' if at least one passage supports the fact even if another passage contradicts the fact.\n" \
                                            "3. Output 'False' if no passage  supports the fact.\n" \
                                            "4. Do not include any additional information and explanations, you must only answer 'True' or 'False'.",
                                      "zh": "你是一名事实核查专家。给定一个原子事实及若干文本段落，" \
                                            "请严格依据段落内容判断该事实为【真】或【假】：" \
                                            "若任一段落直接支持该事实（即使其他段落存在矛盾）" \
                                            "，输出【真】；若无段落支持，输出【假】。仅回答【真】或【假】，无需任何解释。",
                                      "ru": "Вы — эксперт по извлечению и проверке фактов. " \
                                            "Вам предоставлен атомарный факт и список текстовых фрагментов. Ваша задача — определить, является ли атомарный факт " \
                                            "«Верно» или «Неверно» исключительно на основе информации, содержащейся в этих фрагментах.\n"
                                            "Инструкции:\n"
                                            "1. Проверьте, поддерживает ли какой-либо из фрагментов факт напрямую.\n"
                                            "2. Выведите «Верно», если хотя бы один фрагмент подтверждает факт, даже если другой фрагмент противоречит ему.\n"
                                            "3. Выведите «Неверно», если ни один из фрагментов не поддерживает факт.\n"
                                            "4. Не добавляйте никакой дополнительной информации и пояснений — отвечайте строго «Верно» или «Неверно».\n",
                                      "zh1": "You are an expert fact extraction and verification assistant. " \
                                             "You are given an atomic fact and a list of textual passages. Your task is to determine whether the atomic fact " \
                                             "is True or False  based solely on the information in the passages.\n" \
                                             "Instructions:\n" \
                                             "1. Check if any of the passages directly support the atomic fact.\n" \
                                             "2. Output 'True' if at least one passage supports the fact even if another passage contradicts the fact.\n" \
                                             "3. Output 'False' if no passage  supports the fact.\n" \
                                             "4. Do not include any additional information and explanations, you must only answer 'True' or 'False'.",

                                      # "zh": "你是一位专业的事实提取与验证助手。"
                                      #       "你将获得一个原子事实和一组文本段落。你的任务是仅根据段落中的信息，判断该原子事实为‘真’或‘假’。\n" \
                                      #       "指令：1. 检查是否有任何段落直接支持该原子事实。\n" \
                                      #       "2. 如果至少有一个段落支持该事实，即使其他段落与之矛盾，也输出‘True’。\n" \
                                      #       "3. 如果没有任何段落支持该事实，则输出‘False’。\n" \
                                      #       "4. 不要包含任何额外信息或解释，你只能回答‘True’或‘False’。"
                                      }

MULTIFACT_VERIFICATION_SYSTEM_PROMPT = "You are an expert fact verification assistant. You are given:\n" \
                                       "1. The full text of a Wikipedia page on a relevant topic.\n" \
                                       "2. An enumerated list of atomic facts (N facts), each on a new line.\n\n" \
                                       "Your task is to determine whether each fact is supported or refuted by the Wikipedia page.\n\n" \
                                       "Instructions:\n" \
                                       "1. Check if any part of the provided Wikipedia page directly supports the fact.\n" \
                                       "2. Output 'True' if at least one part of the page supports the fact, even if other parts contradict it.\n" \
                                       "3. Output 'False' if no part of the page supports the fact.\n" \
                                       "4. For each fact, you must output exactly a single line with the fact's integer identifier and True/False atomic fact label. The line format is as follows:\n" \
                                       "[fact_id]. [True/False]\n" \
                                       "where [fact_id] is the integer identifier from the input list.\n" \
                                       "5. Do not include any additional explanations, comments, or formatting."
# DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE = """
# Passages:
# <passages>
# Atomic Fact: <atomic_fact>
# True or False?
# """


DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE = {
    "en": """Passages:\n<passages>\nAtomic Fact:\n<atomic_fact>\nTrue or False?\n""",
    "ru": "Текстовые фрагменты:\n<passages>\n"
          "Атомарный факт:\n<atomic_fact>\nВерно или Неверно?\n",
    "zh": "段落：\n<passages>\n原子事实：\n<atomic_fact>\n正确还是错误？",
    "zh1": """Passages:\n<passages>\nAtomic Fact:\n<atomic_fact>\nTrue or False?\n""",
}

DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE_WITH_TOPIC = """
Passages:
<passages>
Atomic Fact's topic/topics: <fact_topic>
Atomic Fact: <atomic_fact>
True or False? 
"""

MULTIFACT_VERIFICATION_QUERY_TEMPLATE = """
Wikipedia page:
<page_content>


Atomic Facts:
<atomic_facts>

Atomic fact labels:
"""

MULTIFACT_VERIFICATION_QUERY_TEMPLATE_WITH_TOPIC = """
Wikipedia page (Page topic: <page_topic>):
<page_content>


Atomic Facts:
<atomic_facts>

Atomic fact labels:
"""


class FactVerificatorSpedUpVLLM(object):
    def __init__(self, vllm_model, model_name, is_bio=False, debug=False,
                 system_prompt=None, query_prompt_template=None, max_tokens: int = 1024, temperature: float = 0.,
                 context_type="db", vllm_tqdm=False, lang: str = "en", lora_request=None):
        assert context_type in ("db", "wikipedia_api")
        self.is_bio = is_bio
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vllm = VLLMGenerator(vllm_model=vllm_model, model_name=model_name, debug=debug, temperature=temperature,
                                  max_tokens=max_tokens)

        self.debug = debug
        if system_prompt is None:
            system_prompt = DEFAULT_VERIFICATION_SYSTEM_PROMPT[lang]
        self.system_prompt = system_prompt
        if query_prompt_template is None:
            query_prompt_template = DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE[lang]
        self.query_prompt_template = query_prompt_template
        self.max_new_tokens = max_tokens
        self.temperature = temperature
        self.context_type = context_type
        self.vllm_tqdm = vllm_tqdm
        self.lora_request = lora_request

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=[self.tokenizer.eos_token]
        )

    def create_messages_single_fact(self, query, passages, topic=None):
        assert self.system_prompt is not None
        assert self.query_prompt_template is not None

        if topic is not None:
            query_prompt_template = DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE_WITH_TOPIC
            query_prompt = query_prompt_template.replace('<atomic_fact>', query)
            if isinstance(topic, list):
                topic_s = ','.join(sorted(topic))
            elif isinstance(topic, str):
                topic_s = topic
            else:
                raise ValueError(f"Unsupported topic type {type(topic)} for {topic}")
            query_prompt = query_prompt.replace("<fact_topic>", topic_s)

        else:
            query_prompt_template = self.query_prompt_template
            query_prompt = query_prompt_template.replace('<atomic_fact>', query)

        context_s = ""
        for i, psg in enumerate(passages):
            s = "Title: {}\nText: {}\n\n".format(psg["title"],
                                                 psg["text"].replace("<s>", "").replace("</s>", ""))
            context_s += s
        query_prompt = query_prompt.replace("<passages>", context_s)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query_prompt}
        ]

        return messages

    def create_messages_multi_fact(self, atomic_facts, context_page, topic=None):
        assert self.system_prompt is not None
        assert self.query_prompt_template is not None

        if topic is not None:
            query_prompt = MULTIFACT_VERIFICATION_QUERY_TEMPLATE_WITH_TOPIC
            if isinstance(topic, list):
                topic_s = ','.join(sorted(topic))
            elif isinstance(topic, str):
                topic_s = topic
            else:
                raise ValueError(f"Unsupported topic type {type(topic)} for {topic}")
            query_prompt = query_prompt.replace("<page_topic>", topic_s)

        else:
            query_prompt = MULTIFACT_VERIFICATION_QUERY_TEMPLATE
            # query_prompt = query_prompt_template.replace('<atomic_fact>', query)
        query_prompt = query_prompt.replace("<page_content>", context_page)
        afs_str = ''.join((f"{i}. {x}\n" for i, x in enumerate(atomic_facts)))
        query_prompt = query_prompt.replace("<atomic_facts>", f"{afs_str}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query_prompt}
        ]

        return messages

    def generate(self, prompts):
        return self.vllm.generate(prompts, use_tqdm=self.vllm_tqdm)
        #, lora_request=self.lora_request)
