# Reducing Omission Errors in Machine Translation with Prompt-Contrastive Decoding

> **Important:**  
> - All Python scripts (`scripts/`, `translation_models/`, and any script ending with .py.) are set up for using prompt-contrastive decoding with **user message** only.
> - For **system prompt** experiments, please use the appropriate notebook in the `notebooks/` directory.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2309.07098-b31b1b.svg)](https://arxiv.org/abs/2309.07098)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<p align="center">
  <img src="logo.png" width="500"/>
</p>

## Overview

This repository builds on and extends [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098)'s codebase to support Llama 3.1 models and prompt-contrastive decoding, as part of Xiaojing Zhang’s master’s thesis at the University of Zurich. The goal is to reduce omission errors in low-resource machine translation with large language models (LLMs) through prompt-based decoding techniques.

## Main Features and Changes

- Support for Llama 3.0 and 3.1, including chat-based prompting
- Implementation of prompt-contrastive decoding
- Adapted scripts and new notebooks for Colab compatibility

## Relation to Original Codebase

All core logic and the original implementation are credited to [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098). This fork was extended and maintained by Xiaojing Zhang for the master’s thesis at the University of Zurich.

This project is based on the source-contrastive and language-contrastive decoding framework as described in [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098):

- **Source-contrastive decoding**: Search for a translation that maximizes P(_Y_|_X_) - λ·P(_Y_|_X'_), where _X'_ is a random source segment. This penalizes hallucinations.
- **Language-contrastive decoding**: Search for a translation that maximizes P(_Y_|_X_,_l_y_) - λ·P(_Y_|_X_,_l_y'_), where _l_y_ is the language indicator for the desired target language, and _l_y'_ the indicator for an undesired language (such as English or the source language). This penalizes off-target translations.
- **Prompt-contrastive decoding** (this work): Search for a translation that maximizes P(_Y_|_X_, _p_pos_) – λ·P(_Y_|_X'_, _p_neg_), where _p_pos_ is the positive prompt that encourages desired translation behavior and _p_neg_ the negative prompt inducing undesired translation behavior such as omissions. This penalizes omissions.

**Main modifications in this repository:**

- **llama.py**:
  1. Added pad token, set padding side, and defined EOS token IDs for Llama 3.1 models
  2. Used role-based chat template and tokenizer's `apply_chat_template` method for Llama 3.1 models
  3. Removed the `PromptTemplate` class as the chat template now handles prompt formatting
  4. Replaced pipeline use (preprocess, forward, and postprocess) as the chat template for Llama 3.1 is a list of dictionaries, but pipeline does not accept it
  5. Made changes to padding side and token, stacked padded tensors into a single batch tensor as input to the model
  6. Added a new parameter `is_prompt_contrastive` to handle contrastive prompts

- **__init__.py**: Added Llama 3 and 3.1 models

- **prompts.py**: New script to handle positive and negative prompts

- **mt_task.py**: Updated the `evaluate` method to handle contrastive prompt pairs

- **run.py**: Added two arguments to handle contrastive prompt pairs

- **utils_run.py** and **utils_llama.py**: updated language codes for FLORES+ dataset

## Folder Structure

- `annotations/`: Manual annotation files (error analysis, omissions)
- `notebooks/`: Notebooks for demos and reproducing thesis results
- `outputs/`: Translation outputs and evaluation results generated in this thesis
- `predictions/`: Original outputs from [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098), for comparison/reference
- `scripts/`: Main experiment scripts and helper utilities
- `tests/`: Unit tests for core modules from [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098)
- `translation_models/`: Model wrappers and utilities (Llama, m2m100, small100)
- `illustration.png`, `logo.png`: Visual assets for documentation/thesis
- `LICENSE`, `README.md`, `requirements.txt`: Repository metadata and setup

## Virtual Environment
##### Set up an virtual environment
- `python3 -m venv venv` for Linux/Mac
### or
- `python -m venv venv` for Windows

##### Activate the virtual environment
- `source venv/bin/activate` for Linux/Mac
- `venv\Scripts\activate` for Windows



## Installation

- `pip install -r requirements.txt`

## Usage

- For prompt-contrastive decoding with user message:  
  Use the Python scripts as described below.

- For prompt-contrastive decoding with system prompt:  
  Please run the relevant notebook in `notebooks/`.


**Example commands**

Source-contrastive decoding with [M2M-100 (418M)](https://arxiv.org/abs/2010.11125) on Asturian–Croatian, with λ_src=0.7:
- `python -m scripts.run --model_path m2m100_418M --language_pairs ast-hr --source_contrastive --source_weight -0.7`

Source-contrastive and language-contrastive decoding with [SMaLL-100](https://arxiv.org/abs/2210.11621) on Pashto–Asturian, with 2 random source segments, λ_src=0.7, λ_lang=0.1, and English and Pashto as contrastive target languages:
- `python -m scripts.run --model_path small100 --language_pairs ps-ast --source_contrastive 2 --source_weight -0.7 --language_contrastive en ps  --language_weight -0.1`

Prompt-contrastive decoding with [Llama 3.1 8B Instruct](https://arxiv.org/abs/2407.21783) on Mongolian-English, with λ_prompt=0.1 and one contrastive prompt pair appended to user message:
- `python -m scripts.run --model_path llama-3.1-8b-instruct --language_pairs mn-en --prompt_contrastive  --prompt_weight -0.1`

Source-contrastive and prompt-contrastive decoding with [Llama 3.1 8B Instruct](https://arxiv.org/abs/2407.21783) on Igbo-English, with 1 random source segment, λ_src=0.7, λ_prompt=0.1:
- `python -m scripts.run --model_path llama-3.1-8b-instruct --language_pairs ig-en --source_contrastive --source_weight -0.7 --prompt_contrastive  --prompt_weight -0.1`

Or run the provided notebook for a full Colab demo.

## Dataset and Models:

- [FLORES-101](https://huggingface.co/datasets/gsarti/flores_101), as in original repo
- [FLORES_Plus](https://huggingface.co/datasets/openlanguagedata/flores_plus). `devtest` section is used for the evaluation.

Multiple models are implemented:

- [M2M-100 (418M)](https://huggingface.co/facebook/m2m100_418M). Use `--model_path m2m100_418M`
- [SMaLL-100](https://huggingface.co/alirezamsh/small100). Use `--model_path small100`
- [Llama 3.1 8B Instruct](https://arxiv.org/abs/2407.21783). Use `--model_path llama-3.1-8b-instruct`


## Evaluation

ChrF2:
```
sacrebleu ref.txt < output.txt --metrics chrf
```


spBLEU:
```
sacrebleu ref.txt < output.txt --tokenize flores101
```


MetricX-23-XL:
Run the provided notebook.


## Reference

```bibtex
@inproceedings{sennrich-etal-2024-mitigating,
      title={Mitigating Hallucinations and Off-target Machine Translation with Source-Contrastive and Language-Contrastive Decoding}, 
      author={Rico Sennrich and Jannis Vamvas and Alireza Mohammadshahi},
      booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
      year={2024}
}
```
