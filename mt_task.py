import logging
import subprocess
import tempfile
import random
import copy
from pathlib import Path
from scripts.utils_run import FLORES101_CONVERT
from sacrebleu import get_source_file
from datasets import load_dataset
from tqdm import tqdm
import os
from translation_models.prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS


class MTTask:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 testset: str,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.language_pair = f"{src_lang}-{tgt_lang}"
        self.testset = testset
        base_out_dir = Path(__file__).parent / "out"
        print(base_out_dir)
        base_out_dir.mkdir(parents=True, exist_ok=True)
        assert base_out_dir.exists()
        self.out_dir = base_out_dir / self.testset
        self.out_dir.mkdir(exist_ok=True)

        self.out_dir = self.out_dir / self.language_pair
        self.out_dir.mkdir(exist_ok=True)
        self.load_converter = FLORES101_CONVERT

    def __str__(self):
        return f"{self.testset}-{self.src_lang}-{self.tgt_lang}"

    def evaluate(self, translation_method: callable, type='direct', source_contrastive=1, source_weight=None,language_contrastive=None, language_weight=None, prompt_contrastive=False, prompt_weight=None) -> Path:

        ## load FLORES dataset
        #source_sentences = load_dataset('gsarti/flores_101',self.load_converter[self.src_lang])['devtest']['sentence']

        ## load FLORES Plus dataset
        dataset = load_dataset('openlanguagedata/flores_plus')
        # Filter the 'dev' split for the target language using iso_639_3
        source_sentences = [
            example['text']
            for example in dataset['devtest']
            if example['iso_639_3'] == self.load_converter[self.src_lang]
        ]

        if type == 'direct':
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            source_sentences=source_sentences,
            )
        elif type == 'contrastive':
            multi_source_sentences = []
            src_weights = [1]
            tgt_langs=[]
            src_langs=[]

            # randomly shuffled input to suppress hallucinations
            if source_contrastive:
                multi_source_sentences.append(source_sentences)
                src_weights.append(1.0)
                tgt_langs.append(self.tgt_lang)
                src_langs.append(self.src_lang)

                for i in range(source_contrastive):
                    shuffled_sentences = copy.copy(source_sentences)
                    random.shuffle(shuffled_sentences)
                    multi_source_sentences.append(shuffled_sentences)
                    src_weights.append(source_weight/source_contrastive)
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)

                print(f"Source Weights (source contrastive): {src_weights}")

            # input with wrong target language indicator to suppress off-target translation
            if language_contrastive:
                multi_source_sentences.append(source_sentences)
                src_weights.append(1.0)
                tgt_langs.append(self.tgt_lang)
                src_langs.append(self.src_lang)

                for offtarget in language_contrastive:
                    # ignore contrastive variants that are identical to true translation direction
                    if offtarget == self.tgt_lang:
                        continue
                    # don't create contrastive variant for src language if language is already listed (avoid duplicates)
                    if offtarget == 'src' and self.src_lang in language_contrastive:
                        continue
                    multi_source_sentences.append(source_sentences)
                    src_weights.append(language_weight)
                    if offtarget == 'src':
                        tgt_langs.append(self.src_lang)
                    else:
                        tgt_langs.append(offtarget)
                    src_langs.append(self.src_lang)

                print(f"Language Weights (language contrastive): {src_weights}")


            # prompt-contrastive: add positive and negative prompts
            if prompt_contrastive:

                # Add positive prompts
                for positive_prompt in POSITIVE_PROMPTS:
                    #print(f"Adding positive prompt: {positive_prompt}")
                    positive_prompts = [
                        f"{src_sent}\n\n{positive_prompt}"
                        for src_sent in source_sentences
                    ]

                    multi_source_sentences.append(positive_prompts)
                    src_weights.append(1.0)  # Positive prompts have weight 1.0
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)

                # Add negative prompts
                for negative_prompt in NEGATIVE_PROMPTS:
                    #print(f"Adding negative prompt: {negative_prompt}")
                    negative_prompts = [
                        f"{src_sent}\n\n{negative_prompt}"
                        for src_sent in source_sentences
                    ]

                    multi_source_sentences.append(negative_prompts)
                    src_weights.append(prompt_weight)  # Negative prompts have weight prompt_weight
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)

                is_prompt_contrastive=True
                print(f"Prompt Weights: {src_weights}")

            else:
                is_prompt_contrastive=False

            translations = []
            for pair in tqdm(list(zip(*multi_source_sentences))):
                translation = translation_method(
                    src_langs=src_langs,
                    tgt_langs=tgt_langs,
                    src_weights=src_weights,
                    multi_source_sentences=pair,
                    is_prompt_contrastive=is_prompt_contrastive,
                    )
                translations.append(translation)
        else:
            raise NotImplementedError

        if type == 'direct':
            file_name = 'direct_devtest'
        elif type == 'contrastive':
            file_name = 'contrastive-{0}-{1}'.format(source_contrastive, source_weight)
            if language_contrastive:
                file_name += "-lang-{0}-{1}".format('+'.join(language_contrastive), language_weight)
            if prompt_contrastive:
                file_name = "contrastive-prompt-general".format(prompt_weight)
        else:
            raise NotImplementedError

        self.out_dir.mkdir(parents=True, exist_ok=True)

        with open(str(self.out_dir)+"/"+file_name+".txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(translations))

        if not os.path.isfile(str(self.out_dir)+"/"+"ref.txt"):
            # target_sentences = load_dataset('gsarti/flores_101', self.load_converter[self.tgt_lang])['devtest']['sentence']
            target_sentences = [
                example['text']
                for example in dataset['devtest']
                if example['iso_639_3'] == self.load_converter[self.tgt_lang]
            ]

            with open(str(self.out_dir) + "/" + "ref.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(target_sentences))

        return Path(f.name)
