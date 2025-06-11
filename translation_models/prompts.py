"""
Defines positive and negative prompts for prompt-contrastive decoding experiments.
These prompts are used to guide large language models to either include or omit specific details in translation,
for the purpose of evaluating and reducing omission errors and untranslated words.

The prompts are grouped by their role:
    - POSITIVE_PROMPTS: Encourage full and accurate translation (minimize omissions).
    - NEGATIVE_PROMPTS: Encourage omissions or untranslated elements (simulate error cases).
"""

# List of positive prompts to encourage the model to include all relevant information
POSITIVE_PROMPTS = [
    # Positive Prompt for omissions
    #"Translate the source text into English accurately, fluently, and fully. Do not omit any information. \nFollow these rules:\n1. Include and translate connectives like 'agbanyeghi' ('however', 'although', or 'while') and 'mana' ('but'), especially when they appear at the beginning of a sentence.\n2. Do not omit proper nouns, especially names of countries, organizations, and places. Translate all proper nouns into English accurately.\n3. Translate temporal expressions and time phrases accurately ('ụnyahụ' to 'yesterday', 'otu izu gara aga' to 'a week ago'). \n4. Include and translate adverbs 'na-emekarị' to 'usually' and 'ọtụtụ mgbe' to 'often'.\n5. Pay special attention to phrases with embedded information, particularly those in commas or parentheses, and sentences beginning with linking words.\n6. Translate subordinate and prepositional clauses fully (starting with 'ọ bụrụ', 'n'ihi na', 'mgbe ole').\n7. When the source text includes a list of items, make sure all elements are included and accurately translated.",
    # Positive Prompt (simple and general prompt)
    "Translate the following source text into English accurately, fluently, and fully.",
    # Positive Prompt (light rule-based)
    #"Translate the following source text into English accurately, fluently, and fully. Follow these rules: \nTranslate all details in the source text, especially connectives, temporal expressions, adverbs, and subordinate clauses.\nTranslate all proper nouns into English accurately (names of organizations, places and people).",
]

# List of negative prompts to encourage the model to omit details or leave words untranslated
NEGATIVE_PROMPTS = [
    # Negative Prompt for untranslated words
    #"Keep all proper nouns in their original Igbo-adapted form. Do not translate names of people, places, or countries into English.",
    # Negative Prompt omissions
    #"Translate the source text into English accurately, fluently, and concisely. Follow these instructions:\n1. Omit details that are not central to the sentence’s meaning.\n2. Omit connective words like 'agbanyeghi' or 'mana'.\n3. Skip words or phrases indicating time such as 'ụnyahụ' or 'otu izu gara aga'.\n4. Omit information in commas or parentheses.\n5. Avoid translating full lists, just translate a few representative items in the lists.",
    # Negative Prompt for omissions (simple and general prompt)
    "Translate the source text into English accurately and briefly, with omissions. Omit all adverbs, connectives, linking words, temporal expressions, prepositional and subordinate clauses.",
]
