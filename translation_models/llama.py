import logging
from typing import Set, List, Union, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessorList

from scripts.utils_run import FLORES101_CONVERT
from translation_models import TranslationModel
from translation_models.m2m100 import EnsembleLogitsProcessor
from translation_models.utils_llama import language_names, one_shot_sentences
from translation_models.prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS


class LLaMaTranslationModel(TranslationModel):

    # Official templates used during instruction tuning of LLaMA
    TEMPLATE_0 = "{src_sent}\n\nTranslate to {tgt_lang}"
    TEMPLATE_1 = "{src_sent}\n\nCould you please translate this to {tgt_lang}?"
    TEMPLATE_2 = "{src_sent}\n\nTranslate this to {tgt_lang}?"
    TEMPLATE_3 = "Translate to {tgt_lang}:\n\n{src_sent}"
    TEMPLATE_4 = "Translate the following sentence to {tgt_lang}:\n{src_sent}"
    TEMPLATE_5 = "How is \"{src_sent}\" said in {tgt_lang}?"
    TEMPLATE_6 = "Translate \"{src_sent}\" to {tgt_lang}?"

    SYSTEM_PROMPT = """You are a machine translation system that translates sentences from {src_lang} to {tgt_lang}. You just respond with the translation, without any additional comments."""

    def __init__(self,
                 model_name_or_path: str,
                 message_template: str = TEMPLATE_0,
                 one_shot: bool = False,
                 padding: str = "before_system_prompt",
                 **kwargs,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', load_in_4bit=True,
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.message_template = message_template
        self.one_shot = one_shot

        assert padding in ["before_system_prompt", "after_system_prompt"], \
            "Padding must be 'before_system_prompt' or 'after_system_prompt'"
        if padding == "before_system_prompt":
            self.tokenizer.padding_side = 'left'
        elif padding == "after_system_prompt":
            self.tokenizer.padding_side = 'right'

        self.padding = padding
        self.src_lang = None
        self.tgt_lang = None

        # Terminators for Llama3 and 3.1
        self.eos_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Add a new pad token
        if self.tokenizer.pad_token is None:
          self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
          self.model.resize_token_embeddings(len(self.tokenizer))
          logging.info("Adding new pad token '[PAD]")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def __str__(self):
        return str(self.model_name_or_path).replace("/", "_")

    @property
    def supported_languages(self) -> Set[str]:
        return {code for code, code3 in FLORES101_CONVERT.items() if code3 in language_names}

    def requires_src_lang(self):
        return True

    def _set_src_lang(self, src_lang: str):
        assert src_lang in self.supported_languages
        self.src_lang = src_lang

    def _set_tgt_lang(self, tgt_lang: str):
        assert tgt_lang in self.supported_languages
        self.tgt_lang = tgt_lang

    def _lang_code_to_name(self, lang_code: str) -> str:
        lang_code3 = FLORES101_CONVERT.get(lang_code, lang_code)
        return language_names[lang_code3]

    @torch.no_grad()
    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 1,
                   num_beams: int = 1,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        if return_score:
            raise NotImplementedError
        if batch_size != 1:
            logging.warning(
                f"Batch size {batch_size} is not supported by LLaMaTranslationModel. Setting batch size to 1.")
            batch_size = 1
        if num_beams != 1:
            logging.warning(f"Beam search is not supported by LLaMaTranslationModel. Setting num_beams to 1.")
            num_beams = 1

        assert self.src_lang is not None
        assert self.tgt_lang is not None
        system_prompt = self.SYSTEM_PROMPT.format(
            src_lang=self._lang_code_to_name(self.src_lang),
            tgt_lang=self._lang_code_to_name(self.tgt_lang),
        )

        if self.one_shot:
            system_prompt += "\n\nExample instruction:\n{instruction}\n\nExample response:\nSure, here's the translation:\n{response}".format(
                instruction=self.message_template.format(
                    src_lang=self._lang_code_to_name(self.src_lang),
                    tgt_lang=self._lang_code_to_name(self.tgt_lang),
                    src_sent=one_shot_sentences[FLORES101_CONVERT.get(self.src_lang, self.src_lang)],
                ),
                response=one_shot_sentences[FLORES101_CONVERT.get(self.tgt_lang, self.tgt_lang)],
            )

        translations = []
        for source_sentence in tqdm(source_sentences):
            message = self.message_template.format(
                src_lang=self._lang_code_to_name(self.src_lang),
                tgt_lang=self._lang_code_to_name(self.tgt_lang),
                src_sent=source_sentence,
            )
            logging.info(message)

            # Prepare messages for Llama3 with roles: system, user, assistant
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Sure, here's the translation:\n"}
            ]

            print(f"Messages sent to the model: \n{messages}")

            # Tokenize messages
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # Generate responses
            outputs = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.pad_token_id,  # Add pad token
                eos_token_id=self.eos_token_ids,
                max_length=1200,  # Max ref length across Flores-101 is 960
                remove_invalid_values=True,
                num_beams=num_beams,
                # Disable sampling
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                **kwargs
            )
            # print(f"Outputs for direct translation: \n{outputs}")

            response = outputs[0][input_ids.shape[-1]:]
            output = self.tokenizer.decode(response, skip_special_tokens=True)
            # print(f"Model's output:\n{output}")
            logging.info(output)
            response_lines = output.replace("Sure, here's the translation:", "").strip().split("\n")
            if not response_lines:
                translation = ""
            else:
                translation = response_lines[0].strip()
            translations.append(translation)
            print(f"Translations:\n{translation}")
        return translations

    def _translate_multi_source(self,
                                multi_source_sentences: List[str],
                                src_langs: List[str],
                                tgt_langs: List[str],
                                src_weights: Optional[List[float]] = None,
                                num_beams: int = 1,
                                is_prompt_contrastive=False,
                                **kwargs,
                                ) -> str:
        assert len(multi_source_sentences) == len(src_langs) == len(tgt_langs)
        if src_weights is not None:
            assert len(src_weights) == len(multi_source_sentences)
        if num_beams != 1:
            logging.warning(f"Beam search is not supported by LLaMaTranslationModel. Setting num_beams to 1.")
            num_beams = 1

        prompts = []
        for src_sent, src_lang, tgt_lang in zip(multi_source_sentences, src_langs, tgt_langs):
            system_prompt = self.SYSTEM_PROMPT.format(
                src_lang=self._lang_code_to_name(src_lang),
                tgt_lang=self._lang_code_to_name(tgt_lang),
            )

            if self.one_shot:
                system_prompt += "\n\nExample instruction:\n{instruction}\n\nExample response:\nSure, here's the translation:\n{response}".format(
                    instruction=self.message_template.format(
                        src_lang=self._lang_code_to_name(src_lang),
                        tgt_lang=self._lang_code_to_name(tgt_lang),
                        src_sent=one_shot_sentences[FLORES101_CONVERT.get(src_lang, src_lang)],
                    ),
                    response=one_shot_sentences[FLORES101_CONVERT.get(tgt_lang, tgt_lang)],
                )

            if is_prompt_contrastive:
                # Split src_sent into original source sentence and positive/negative prompt
                if "\n\n" in src_sent:
                    original_source, prompt = src_sent.split("\n\n", 1)
                else:
                    original_source, prompt = src_sent, ""

                # Create user message with original source sentence
                message = self.message_template.format(
                    src_lang=self._lang_code_to_name(src_lang),
                    tgt_lang=self._lang_code_to_name(tgt_lang),
                    src_sent=original_source,
                )

                # Append positive/negative prompt to the message
                full_message = f"{prompt}\n\n{message}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_message},
                    {"role": "assistant", "content": "Sure, here's the translation:\n"}
                ]

            else:
                # Use TEMPLATE_0 directly with source_contrastive and language_contrastive
                message = self.message_template.format(
                    src_lang=self._lang_code_to_name(src_lang),
                    tgt_lang=self._lang_code_to_name(tgt_lang),
                    src_sent=src_sent,
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Sure, here's the translation:\n"}
                ]

            prompts.append(messages)
            print(f"Prompts: \n{prompts}")

        # Tokenize messages
        input_ids = []
        attention_masks = []

        for messages in prompts:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                padding=False,
                return_tensors='pt',
            )
            input_ids.append(inputs[0])
            attention_masks.append(torch.ones_like(inputs[0]))

        # Manual padding
        max_len = max(tensor.size(0) for tensor in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        pad_token_id = self.tokenizer.pad_token_id

        for ids, mask in zip(input_ids, attention_masks):
            pad_len = max_len - ids.size(0)
            if self.padding == "before_system_prompt":
                padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
                padded_mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            else:
                padded_ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        # Batch inputs
        input_ids = torch.stack(padded_input_ids).to(self.model.device)
        attention_mask = torch.stack(padded_attention_masks).to(self.model.device)

        #print(f"Input Ids: \n{input_ids}")
        #print(f"Attention Masks: \n{attention_mask}")

        # Generate with logits processor for contrastive decoding
        logits_processor = LogitsProcessorList([
            EnsembleLogitsProcessor(num_beams=num_beams, source_weights=src_weights),
        ])

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1200,
            logits_processor=logits_processor,
            remove_invalid_values=True,
            # Disable sampling
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            **kwargs,
        )
        #print(f"Outputs by Llama 3.1: \n{outputs}")

        response = outputs[0][input_ids.shape[-1]:]
        output = self.tokenizer.decode(response, skip_special_tokens=True)

        response_lines = output.replace("Sure, here's the translation:", "").strip().split("\n")

        if not response_lines:
            translation = ""
        else:
            translation = response_lines[0].strip()
        print(f"Translation: \n{translation}")
        return translation

