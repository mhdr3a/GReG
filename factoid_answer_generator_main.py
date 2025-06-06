# Credit for Strict (Sacc) and Lenient (Lacc) Accuracy Prompts for OpenAI LLMs:
# https://github.com/SamyAteia/bioasq

import argparse
import re
from unidecode import unidecode
import json
from tqdm import trange
import openai
from openai import OpenAI
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from termcolor import colored
import os
import torch
from copy import deepcopy
import string
import collections
import ftfy
from typing import Tuple, List
import spacy
import math
import random

def improve_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 0
improve_reproducibility(seed)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_answer_tag(text):
        return re.sub(r'^<\s*answer\s*>\s*[:：]?\s*', '', text, flags=re.IGNORECASE)

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = remove_answer_tag(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def correct_json_list(json_string):
    corrected_string = re.sub(r'^.*?\[', '[', json_string, flags=re.DOTALL)
    corrected_string = re.sub(r'\].*$', ']', corrected_string, flags=re.DOTALL)
    try:
        _ = json.loads(corrected_string)
    except json.JSONDecodeError:
        return '[]'
    return corrected_string

def highlight_ngrams(sentence, ngrams_to_highlight=[], color='red', highlight_all=True):
    if highlight_all:
        return colored(sentence, color)
    ngrams_to_highlight = sorted(ngrams_to_highlight, key=lambda x: len(x.split()), reverse=True)
    for ngram in ngrams_to_highlight:
        sentence = sentence.replace(ngram, colored(ngram, color))
    return sentence

def compute_semantic_similarity(reference, generated):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        # Access the first element of the tuple which contains the hidden states
        hidden_states = outputs[0]
        return hidden_states.mean(dim=1).detach().numpy()

    ref_embedding = get_embedding(reference)
    gen_embedding = get_embedding(generated)

    cosine_similarity = np.dot(ref_embedding, gen_embedding.T) / (np.linalg.norm(ref_embedding) * np.linalg.norm(gen_embedding))
    return cosine_similarity.item()

def normalize(str1):
    str1 = str(str1)
    str1_normalized = re.sub(r'[^\w\d]', '', unidecode(re.sub(r'\b(a|an|the)\b', '', str1, flags=re.IGNORECASE)).lower())
    return str1_normalized

def compare(str1, str2):
    return str1 == str2

def compare_with_list(list1, str1):
    return any(list(map(lambda x: x == str1, list1)))

def estimate_cost(total_samples, fixed_instructions_msg, model, questions, max_output_tokens_per_sample, contexts=None, word_to_token_ratio=1.33, include_contexts=True, question_key='question', using_batch_api=True):
    fixed_instructions_tokens = word_to_token_ratio * len(fixed_instructions_msg.replace('\n', ' ').strip().split())
    total_context_tokens = word_to_token_ratio * sum(list(map(lambda x: sum(list(map(lambda y: len(y.replace('\n', ' ').strip().split()), x))), contexts))) if include_contexts else 0
    total_input_tokens = word_to_token_ratio * sum(list(map(lambda q: len(q[question_key].replace('\n', ' ').strip().split()), questions))) + total_samples * fixed_instructions_tokens + total_context_tokens
    total_output_tokens = total_samples * max_output_tokens_per_sample
    cost_per_1M_tokens_input = 5 if model == 'gpt-4o' else (.5 if model == 'gpt-3.5-turbo' else (10 if model == 'gpt-4-turbo' else (.600 if model == 'gpt-4o-mini' else (1.5 if model == 'gpt-3.5-turbo-instruct' else None))))
    cost_per_1M_tokens_output = 20 if model == 'gpt-4o' else (1.5 if model == 'gpt-3.5-turbo' else (30 if model == 'gpt-4-turbo' else (2.40 if model == 'gpt-4o-mini' else (2.0 if model == 'gpt-3.5-turbo-instruct' else None))))

    total_tokens, estimated_cost = total_input_tokens + total_output_tokens, cost_per_1M_tokens_input * (total_input_tokens / 1000000) + cost_per_1M_tokens_output * (total_output_tokens / 1000000)

    return (.5 if using_batch_api else 1) * estimated_cost, total_tokens

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def read_json(filepath):
    data = []
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def compute_strict_accuracy(exact_answer, generated_answers):
    return exact_answer == generated_answers[0] if len(generated_answers) > 0 else False

def compute_lenient_accuracy(exact_answer, generated_answers):
    return exact_answer in generated_answers

class InvalidFileExtensionError(Exception):
    def __init__(self, extension, message="Invalid file extension"):
        self.extension = extension
        self.message = f"{message}: {extension}"
        super().__init__(self.message)

def convert_to_question_dict(list_of_dicts):
    question_dict = {}
    for item in list_of_dicts:
        question = item.get("question")
        if question:
            # Use the original dict as the value but keep it as is
            question_dict[question] = item
    return question_dict

def evaluate_saved_file(opt):
    filepath = opt.data
    filepath_org = opt.data_org
    data = read_jsonl(filepath)
    data_org = convert_to_question_dict(read_jsonl(filepath_org))
    n = len(data)
    metrics = opt.metrics # 0: Exact Match and ROUGE | 1: Lenient and Strict Accuracy
    # file_metrics = int(filepath.split('/')[-1].split('_')[4])
    # assert file_metrics == metrics, "metrics argument does not match the saved file's metrics."
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Only works if metrics = 0
    EM = 0; precision = 0; recall = 0; fmeasure = 0; sem_sim = 0 # Only works if metrics = 0
    Sacc = 0; Lacc = 0 # Only works if metrics = 1
    print(f'\n# of samples: {n}')
    for i in trange(n):
        datum = data[i]
        # answer = datum['gold_answer']
        answers = [data_org[data[i]['question']]['answer']]
        answers.extend(data_org[data[i]['question']]['answer_aliases'])
        assert len(answers) > 0 and type(answers) == list
        generated_answer = datum['pred_answer']
        mets = evaluate_factoid_answers(metrics, answers, generated_answer, scorer)
        EM += mets[0]; precision += mets[1]; recall += mets[2]; fmeasure += mets[3]; sem_sim += mets[4]; Sacc += mets[5]; Lacc += mets[6]
    if metrics == 0:
        acc = round(EM/n, 3); pre = round(precision/n, 3); rec = round(recall/n, 3); f1 = round(fmeasure/n, 3); ss = round(sem_sim/n, 3)
        print(f'\nEM: {EM}')
        print(f"Accuracy: {acc}")
        print(f"Precision: {pre}")
        print(f"Recall: {rec}")
        print(f"F-1: {f1}")
        print(f"Sem-Sim: {ss}")
    else:
        Sa = round(Sacc/n, 3); La = round(Lacc/n, 3)
        print(f'\nSacc: {Sa}')
        print(f"Lacc: {La}")

def read_data(filepath, ext):
    data = []
    if ext == 'json':
        data = read_json(filepath)
    elif ext == 'jsonl':
        data = read_jsonl(filepath)
    else:
        raise InvalidFileExtensionError(ext)
    id_key = None
    if 'id' in data[0]:
        id_key = 'id'
    elif '_id' in data[0]:
        id_key = '_id'
    n = len(data)
    return data, id_key, n

def add_contexts(full_prompt, contexts, top_k, include_titles):
    for j in range(top_k):
        title = contexts[j]['title']; text = contexts[j]['text']
        tmp = "\nText: "
        full_prompt += f"{j + 1}. {f'Title: {title}{tmp}' if include_titles else ''}{text}"
        full_prompt += '\n'
        if j != top_k - 1:
            full_prompt += '\n'
    return full_prompt

# def openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
#     openai.api_key = api_key
#     response = openai.completions.create(
#         model=model,
#         messages=[{
#             "role": "user",
#             "content": full_prompt
#         }],
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=model_top_p,
#     )
#     return response.choices[0].message.content

# def openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
#     import openai
#     openai.api_key = api_key

#     response = openai.completions.create(
#         model=model,
#         prompt=full_prompt,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=model_top_p,
#     )
#     return response.choices[0].text.strip()


import time
import openai
from openai import RateLimitError

def openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
    openai.api_key = api_key

    while True:
        response = openai.chat.completions.create(
            model=model,
            # prompt=full_prompt,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=model_top_p,
        )
        # return response.choices[0].text.strip() 
        return response.choices[0].message.content.strip()


def evaluate_factoid_answers(metrics, answer, generated_answer, scorer):
    from typing import Tuple, List

    def normalize_to_string_list(value):
        if isinstance(value, bool):
            return ["true" if value else "false"]
        elif isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return [
                "true" if v is True else "false" if v is False else str(v)
                for v in value
            ]
        else:
            return [str(value)]

    answer = normalize_to_string_list(answer)
    generated_answer = "true" if generated_answer is True else "false" if generated_answer is False else str(generated_answer)

    EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc = 0, 0, 0, 0, 0, 0, 0

    if metrics == 0 or metrics == 2:  # Exact Match (EM) and ROUGE
        if len(answer) == 1:  # Treat as MultihopQA
            answer_str = ftfy.fix_text(answer[0])
            generated_answer = ftfy.fix_text(generated_answer)
            EM = compute_exact(answer_str, generated_answer)
            scores = scorer.score(answer_str, generated_answer)['rougeL']
            precision, recall, fmeasure = scores.precision, scores.recall, scores.fmeasure
            fmeasure = compute_f1(answer_str, generated_answer)
        else:  # List of answers (NQ or TQA)
            ground_truth_answers = [ftfy.fix_text(e) for e in answer]
            generated_answer = ftfy.fix_text(generated_answer)
            assert isinstance(generated_answer, str)
            assert isinstance(ground_truth_answers, (Tuple, List))
            EM = int(metric_max_over_ground_truths(compute_exact, generated_answer, ground_truth_answers))
            fmeasure = metric_max_over_ground_truths(compute_f1, generated_answer, ground_truth_answers)

    elif metrics == 1:  # Strict and Lenient Accuracy
        factoids = json.loads(correct_json_list(generated_answer))
        if len(answer) == 1:  # MultihopQA
            Sacc = compute_strict_accuracy(normalize(answer[0]), list(map(normalize, factoids)))
            Lacc = compute_lenient_accuracy(normalize(answer[0]), list(map(normalize, factoids)))
        else:  # NQ or TQA
            Saccs, Laccs = [], []
            for ans in answer:
                Saccs.append(compute_strict_accuracy(normalize(ans), list(map(normalize, factoids))))
                Laccs.append(compute_lenient_accuracy(normalize(ans), list(map(normalize, factoids))))
            Sacc, Lacc = max(Saccs), max(Laccs)

    if metrics == 2:  # Semantic Similarity
        if len(answer) == 1:
            sem_sim = compute_semantic_similarity(answer[0], generated_answer)
        else:
            sem_sim = max([compute_semantic_similarity(ans, generated_answer) for ans in answer])

    return EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc


# def evaluate_factoid_answers(metrics, answer, generated_answer, scorer):
#     assert type(answer) == str or type(answer) == list
#     EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc = 0, 0, 0, 0, 0, 0, 0
#     if metrics == 0 or metrics == 2: # Exact Match (EM) and ROUGE
#         if type(answer) == str: # for MultihopQA datasets
#             # EM = compare(normalize(answer), normalize(generated_answer))
#             answer = ftfy.fix_text(answer)
#             generated_answer = ftfy.fix_text(generated_answer)
#             EM = compute_exact(answer, generated_answer)
#             scores = scorer.score(answer, generated_answer)['rougeL']
#             precision = scores.precision; recall = scores.recall; fmeasure = scores.fmeasure
#             fmeasure = compute_f1(answer, generated_answer)
#         else: # for NQ and TQA
#             # EM = compare_with_list(list(map(normalize, answer)), normalize(generated_answer))
#             ground_truth_answers = [ftfy.fix_text(e) for e in answer]
#             generated_answer = ftfy.fix_text(generated_answer)
#             # precisions, recalls, fmeasures = [], [], []
#             # for ans in answer:
#             #     scores = scorer.score(ans, generated_answer)['rougeL']
#             #     precisions.append(scores.precision); recalls.append(scores.recall); fmeasures.append(scores.fmeasure)
#             # precision = max(precisions); recall = max(recalls); fmeasure = max(fmeasures)
#             assert isinstance(generated_answer, str)
#             assert isinstance(ground_truth_answers, (Tuple, List))
#             exact_scores = metric_max_over_ground_truths(compute_exact, generated_answer, ground_truth_answers)
#             f1_scores = metric_max_over_ground_truths(compute_f1, generated_answer, ground_truth_answers)
#             EM = int(exact_scores)
#             fmeasure = f1_scores
#     elif metrics == 1: # Sacc and Lacc (Strict and Lenient Accuracy)
#         factoids = json.loads(correct_json_list(generated_answer))
#         if type(answer) == str: # for MultihopQA datasets
#             Sacc = compute_strict_accuracy(normalize(answer), list(map(normalize, factoids)))
#             Lacc = compute_lenient_accuracy(normalize(answer), list(map(normalize, factoids)))
#         else: # for NQ and TQA
#             Saccs, Laccs = [], []
#             for ans in answer:
#                 Sacc = compute_strict_accuracy(normalize(ans), list(map(normalize, factoids)))
#                 Lacc = compute_lenient_accuracy(normalize(ans), list(map(normalize, factoids)))
#                 Saccs.append(Sacc); Laccs.append(Lacc)
#             Sacc = max(Saccs); Lacc = max(Laccs)
#     if metrics == 2: # Semantic Similarity
#         if type(answer) == str: # for MultihopQA datasets
#             sem_sim = compute_semantic_similarity(answer, generated_answer)
#         else: # for NQ and TQA
#             sem_sims = []
#             for ans in answer:
#                 sem_sim = compute_semantic_similarity(ans, generated_answer)
#                 sem_sims.append(sem_sim)
#             sem_sim = max(sem_sims)
#     return EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc





def meta_generator(model, tokenizer, question, max_tokens, temperature, top_p, top_k, contexts=None):
    full_prompt = meta_get_formatted_input(question, top_k, contexts)
    bos_token = tokenizer.bos_token or ""
    tokenized_prompt = tokenizer(bos_token + full_prompt, return_tensors="pt").to(model.device)

    terminators = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [50256]

    if temperature == 0:
        outputs = model.generate(
            input_ids=tokenized_prompt.input_ids,
            attention_mask=tokenized_prompt.attention_mask,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            input_ids=tokenized_prompt.input_ids,
            attention_mask=tokenized_prompt.attention_mask,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    for stop_token in ["Human", "Assistant", "You are an AI assistant"]:
        index = decoded.find(stop_token)
        if index != -1:
            decoded = decoded[:index].strip()

    return decoded



# import re

# def meta_generator(model, tokenizer, question, max_tokens, temperature, top_p, top_k, contexts=None):
#     full_prompt = meta_get_formatted_input(question, top_k, contexts)
#     # bos_token = tokenizer.bos_token or "<|im_start|>"  
#     # full_input = bos_token + full_prompt
#     full_input = full_prompt
#     tokenized_prompt = tokenizer(full_input, return_tensors="pt").to(model.device)
#     terminators = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [50256]  

#     if temperature == 0:  
#         outputs = model.generate(
#             input_ids=tokenized_prompt.input_ids,
#             attention_mask=tokenized_prompt.attention_mask,
#             max_new_tokens=max_tokens,
#             eos_token_id=terminators,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     else:
#         outputs = model.generate(
#             input_ids=tokenized_prompt.input_ids,
#             attention_mask=tokenized_prompt.attention_mask,
#             max_new_tokens=max_tokens,
#             eos_token_id=terminators,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p
#         )

#     response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
#     decoded = tokenizer.decode(response, skip_special_tokens=True).strip()

#     return decoded




def meta_get_formatted_input(question, top_k, contexts, include_titles=True):
    
    system = "System: This is a chat between a user and an artificial intelligence assistant."
    if contexts is None:
        #No C-O-T##
        instruction = (
            "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>."
            # "You are a helpful assistant skilled in answering True/False Multi-hop questions, your task is to answer the given question after <Question>.."
            # "You must use logical reasoning to arrive at the best possible answer\n"
            "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
            "In case of referring to a date or specific place, just name the date or place.\n"
            "The answer must be concise (2–3 words max), factoid-style.\n"  
            # "The answer must be concise **True** or **False***, nothing more.\n"
            "Here are some examples:\n\n"

            "Example 1:\n"
            "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
            "So, the answer is: Little Richard\n\n"
            # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
            # "So, the answer is: False\n\n"

            "Example 2:\n"
            "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
            "So, the answer is: Chinua Achebe\n\n"
            # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
            # "So, the answer is: True\n\n"

            "Example 3:\n"
            "<Question>: 'Remember Me Ballin’' is a CD single by Indo G that features an American rapper born in what year?\n"
            "So, the answer is: 1979\n\n"
            # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
            # "So, the answer is: True\n\n"

            )
            
        # conversation = f"User: {instruction}\n<Question>: {question}\n\nAssistant:"
        conversation = f"User:{instruction}\n<Question>: {question}\n\n"
        formatted_input = f"{system}\n\n{conversation}Assistant:\nSo, the answer is:"

        print(formatted_input)

        # ##C-O-T##
        # instruction = (
        #     # "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>.\n"
        #     "You are a helpful assistant skilled in answering True/False Multi-hop questions, your task is to answer the given question after <Question>."
        #     "You Must First Think about the question **step-by-step**, give your thoughts, and then answer the question after <Question>.\n"
        #     # "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
        #     # "In case of referring to a date or specific place, just name the date or place.\n"
        #     # "The answer must be concise (2–3 words max), factoid-style, and based strictly on the provided knowledge. Your answer should be after <Answer>\n"
        #     "The answer must be concise ***True or False***, nothing more.\n"
        #     "**If you DO NOT know the answer, DO NOT generate anything.**\n"
        #     "IMPORTANT: Output ONLY the final answer. Do NOT include the **thought** process or any other prefixes in your final response.\n\n"
        #     "Here are some examples:\n\n"

        #     # "Here are some examples:\n\n"
        #     "Example 1:\n"
        #     # "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
        #     # "<Thought>: Modern Records was an R&B label with artists including Etta James, Joe Houston, Little Richard, Ike & Tina Turner, and John Lee Hooker in the 1950s and 1960s. The given knowledge confirms that Little Richard was an American musician, singer, actor, and songwriter born on December 5, 1932, and worked with Modern Records\n"
        #     # "<Answer>: Little Richard\n\n"
        #     "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
        #     "Let’s think step by step."
        #     "<Thought>: Laxatives are substances that loosen stools and increase bowel movements. People take laxatives to treat and/or prevent constipation.\n"
        #     "So, the answer is: False\n\n"

        #     "Example 2:\n"
        #     # "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
        #     # "<Thought>: Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Since Chinua Achebe had four different jobs while Rachel Carson had three, the answer is Chinua Achebe.\n"
        #     # "<Answer>: Chinua Achebe\n\n"
        #     "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
        #     "Let’s think step by step."
        #     "<Thought>: Durian has a pungent odor that many people describe as being similar to feet and onions. Unpleasant smells can make people feel nauseous.\n"
        #     "So, the answer is: True\n\n"

        #     "Example 3:\n"
        #     # "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
        #     # "<Thought>: Remember Me Ballin’ is a CD single by Indo G that features Gangsta Boo. The given knowledge states that Gangsta Boo, whose real name is Lola Mitchell, is an American rapper born in 1979.\n"
        #     # "<Answer>: 1979\n"
        #     "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
        #     "Let’s think step by step."
        #     "<Thought>: Monty Python and the Holy Grail was a famous film about King Arthur. In Monty Python and the Holy Grail, swallows are mentioned several times.\n"
        #     "So, the answer is: True\n\n"
            
        #     "Let’s think step by step:"

        # )
        # conversation = f"User: {instruction}\n<Question>: {question}\n\nAssistant:"
        # conversation = f"User:{instruction}\n<Question>: {question}\n\n"
        # formatted_input = f"{system}\n\n{conversation}\nAssistant:\nSo, the answer is:"
        # print(formatted_input)

    else:
        # C-O-T##
        instruction = (
            "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question> based on the provided knowledge enclosed within <doc> and </doc> tags.\n"
            # "You are a helpful assistant skilled in answering True/False Multi-hop questions given the provided knowledge, your task is to answer the given question after <Question>."
            # "You must refer to this knowledge when answering the question.\n"
            "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
            "In case of referring to a date or specific place, just name the date or place.\n"
            "If the knowledge contains the answer, give a **concise, factoid-style answer (2–3 words max)**.\n"
            # "The answer must be concise ***True or False***, nothing more.\n"
            # "If the knowledge does not contain the answer, **do not generate anything**.\n"
            "**IMPORTANT: Do NOT include any thought process, explanation, or reasoning. Only return the final answer after <Answer>.**\n"
            

            "Here are some examples:\n\n"

            "Example 1:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
            "So, the answer is: Little Richard\n\n"
            # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
            # "So, the answer is: False\n\n"

            "Example 2:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
            "So, the answer is: Chinua Achebe\n\n"
            # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
            # "So, the answer is: True\n\n"

            "Example 3:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
            "So, the answer is: 1979\n\n"
            # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
            # "So, the answer is: True\n\n"

            # "Now your question is:\n"

        )

        all_contexts = ""
        for j in range(top_k):
            title = contexts[j]['title']
            text = contexts[j]['text']
            doc_text = f"Title: {title}\nText: {text}" if include_titles else text
            all_contexts += f"{doc_text}\n\n"

        doc_section = f"<doc>\n{all_contexts.strip()}\n</doc>"
        formatted_input = (
            f"{system}\n\n"
            f"User:{instruction}\n\n"
            f"Now use the given knowledge below to answer the question.\n\n"
            f"{doc_section}\n\n"
            f"<Question>: {question}\nAssistant:\nSo, the answer is:"
        )
        print(formatted_input)

        
    return formatted_input


    # C-O-T##

    # else:

    #     instruction = (

    #         # "You are a helpful assistant skilled in answering **True/False Multi-hop** questions based on the provided knowledge. "
    #         "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>."
    #         "Your task is to answer the given question (enclosed within <Question>) using only the information inside <doc> tags.\n"
    #         "You **must** think through the information **step by step**, providing your internal reasoning before answering.\n"
    #         "If the provided knowledge contains relevant information, use it to generate a logical, multi-step thought process. "
    #         "If the knowledge does not contain an explicit answer, reason independently and provide your best judgment.\n"
    #         "Your final response must be a single, concise answer: either ***True*** or ***False*** — nothing else.\n"
    #         "**IMPORTANT:** Your output must ONLY contain the final answer. Do NOT include your thoughts, explanations, or any prefixes in the final output.\n\n"

    #         "Here are some examples:\n\n"

        

    #         # "Example 1:\n"
    #         # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
    #         # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
    #         # "Let’s think step by step.\n"
    #         # "<Thought>: Laxatives are substances that loosen stools and increase bowel movements. People take laxatives to treat and/or prevent constipation.\n"
    #         # "So, the answer is: False\n\n"

    #         # "Example 2:\n"
    #         # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
    #         # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
    #         # "Let’s think step by step.\n"
    #         # "<Thought>: Durian has a pungent odor that many people describe as being similar to feet and onions. Unpleasant smells can make people feel nauseous.\n"
    #         # "So, the answer is: True\n\n"


    #         # "Example 3:\n"
    #         # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
    #         # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
    #         # "Let’s think step by step.\n"
    #         # "<Thought>: Monty Python and the Holy Grail was a famous film about King Arthur. In Monty Python and the Holy Grail, swallows are mentioned several times.\n"
    #         # "So, the answer is: True\n\n"

    #         "Example 1:\n"
    #         "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
    #         "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
    #         "Let's think step-by-step:\n"
    #         "<Thought>: Modern Records was an R&B label with artists including Etta James, Joe Houston, Little Richard, Ike & Tina Turner, and John Lee Hooker in the 1950s and 1960s. The given knowledge confirms that Little Richard was an American musician, singer, actor, and songwriter born on December 5, 1932, and worked with Modern Records.\n"
    #         "So, the answer is: Little Richard\n"


    #         "Example 2:\n"
    #         "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
    #         "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
    #         "Let's think step-by-step:\n"
    #         "<Thought>: Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Since Chinua Achebe had four different jobs while Rachel Carson had three, the answer is Chinua Achebe.\n"
    #         "So, the answer is: Chinua Achebe\n"


    #         "Example 3:\n"
    #         "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
    #         "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
    #         "Let's think step-by-step:\n"
    #         "<Thought>: Remember Me Ballin’ is a CD single by Indo G that features Gangsta Boo. The given knowledge states that Gangsta Boo, whose real name is Lola Mitchell, is an American rapper born in 1979.\n"
    #         "So, the answer is: 1979\n\n"
                
        
    #         "Let’s think step by step:"
        
    #     )

    #     all_contexts = ""
    #     for j in range(top_k):
    #         title = contexts[j]['title']
    #         text = contexts[j]['text']
    #         doc_text = f"Title: {title}\nText: {text}" if include_titles else text
    #         all_contexts += f"{doc_text}\n\n"

    #     doc_section = f"<doc>\n{all_contexts.strip()}\n</doc>"
    #     formatted_input = (
    #         f"{system}\n\n"
    #         f"User:{instruction}\n\n"
    #         f"Now use the given knowledge below to answer the question. Internally reason step-by-step, but only output the final answer, nothing else.\n\n"
    #         f"{doc_section}\n\n"
    #         f"<Question>: {question}\nAssistant:\nSo, the answer is:")
    #     print(formatted_input)

    # return formatted_input

    
def get_entities(sentence, nlp):
    entities = []
    doc = nlp(sentence)
    for ent in doc.ents:
        entities.append(ent.text)
    entities = list(set(entities))
    return list(map(remove_punctuation, entities))

def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:  # To avoid log(0)
            entropy -= p * math.log2(p)
    return entropy

def get_token_level_entropies(token_probabilities):
    entropies = []
    for probabilities in token_probabilities:
        entropy = calculate_entropy(probabilities)
        entropies.append(entropy)
    return entropies

def get_word_level_entropies(tokens, token_entropies):
    def combine_entropies(entropies):
        return max(entropies)
    words = []
    word_entropies = []
    current_word = ""
    current_entropies = []

    for token, entropy in zip(tokens, token_entropies):
        if token.startswith(' ') and current_word:
            words.append(current_word)
            word_entropies.append(combine_entropies(current_entropies))
            current_word = token.strip()
            current_entropies = [entropy]
        else:
            current_word += token
            current_entropies.append(entropy)

    if current_word:
        words.append(current_word)
        word_entropies.append(combine_entropies(current_entropies))

    return words, word_entropies

def remove_punctuation(input_string):
    pattern = f"[{re.escape(string.punctuation)}]"
    normalized_string = re.sub(pattern, '', input_string)
    return normalized_string

def generate_ngrams(words, n):
    words = list(map(remove_punctuation, words))
    ngrams_with_indices = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        indices = list(range(i, i+n))
        ngrams_with_indices.append((ngram, indices))
    return ngrams_with_indices

def locate_ngrams(words, n, output_entities):
    sentence_ngrams_with_indices = generate_ngrams(words, n)
    found_ngrams_with_indices = [(ngram, indices) for ngram, indices in sentence_ngrams_with_indices if ngram in list(map(remove_punctuation, output_entities))]
    return found_ngrams_with_indices

def combine_dependent_words_entropies(words, word_entropies, output_entities, max_window_size=5):
    dependent_words = [[] for _ in range(len(words))]
    for n in range(1, max_window_size + 1):
        found_ngrams = locate_ngrams(words, n, output_entities)
        for ngram in found_ngrams:
            ents = []
            for i in ngram[1]:
                ents.append(word_entropies[i])
                max_ents = max(ents)
            for i in ngram[1]:
                word_entropies[i] = max_ents
                dependent_words[i].append(ngram[0])
    return word_entropies, dependent_words

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    # dataset = filepath.split('/')[-3]
    dataset = opt.dataset
    # answers_path = opt.answers
    # if dataset not in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA']:
    #     dataset = 'IIRC'
    # retriever = filepath.split('/')[-1].split('_')[-1].split('.')[0]
    # if retriever != 'nq' or retriever != 'tqa':
    #     retriever = 'nq'
    # template_temperature = 'high' if filepath.split('/')[-1].count('_') == 5 else 'low'
    # assert retriever in ['nq', 'tqa']
    # percentile = filepath.split('/')[-2].split('_')[-1]
    # if percentile not in ['0', '50', '100', 'x']:
    #     percentile = '100'
    assert dataset in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA', 'StrategyQA']
    
    data, id_key, n = read_data(filepath, ext)
    # answers_data = read_jsonl(answers_path)

    top_k = opt.top_k
    assert 0 <= top_k <= 10
    query_type = opt.query_type
    if top_k > 0:
        assert query_type in ['question', 'GAR', 'GAR_T0.5', 'GAR_T1.0', 'GAR_T1.5', 'long_T0', 'long_T0_GOLD', 'long_T0_LLM', 'long_T0.5', 'long_T0.5_GOLD', 'long_T0.5_LLM', 'long_T1.0', 'long_T1.0_GOLD', 'long_T1.0_LLM', 'long_T1.5', 'long_T1.5_GOLD', 'long_T1.5_LLM', 'long_T0_LLM_long_T1.5', 'mixed_median', 'mixed_median_GAR', 'mixed_median_GAR_GOLD', 'mixed_median_GOLD', 'mixed_median_LLM_max', 'mixed_median_GAR_LLM_min', 'mixed_median_LLM_min', 'mixed_hallu', 'mixed_hallu_GOLD', 'mixed_median_GOLD_v2', 'mixed_median_LLM_min_v2']
    if query_type in [None, 'question']:
        question_key = 'question_org'
    else:
        question_key = 'question_org'
    # answer_keys = ['answer', 'answer_aliases']
    answer_keys = ['answer']
    
    model = opt.model
    assert model in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3', 'llama-3.1', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct', 'vicuna', 'qwen']
    if model == 'llama-3':
        model_id = "nvidia/Llama3-ChatQA-1.5-8B"
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        meta_tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model == 'llama-3.1':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        meta_tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model == 'vicuna':
        model_id = "lmsys/vicuna-13b-v1.5"
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        meta_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        meta_tokenizer.pad_token = meta_tokenizer.eos_token
    elif model == 'qwen':
        model_id = "Qwen/Qwen-7B"
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        meta_tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, trust_remote_code=True
        )
        meta_tokenizer.pad_token = meta_tokenizer.eos_token    
    v = opt.verbose
    assert v == 0 or v == 1
    # LAG = opt.LAG
    # assert LAG in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct']
    max_tokens = opt.max_tokens
    assert max_tokens > 0
    # temperatures = [float(t) for t in opt.temperature.split(',')]
    # for temp in temperatures:
    #     assert 0 <= temp <= 2, f"Temperature {temp} is out of valid range (0, 2)"
    temperature = opt.temperature
    assert 0 <= temperature <= 2
    model_top_p = opt.top_p
    assert .1 <= model_top_p <= 1
    api_key = opt.api_key
    assert api_key != None if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct'] else True, "Please specify your OpenAI API key."
    
    include_titles = opt.include_titles
    assert include_titles == 0 or include_titles == 1
    metrics = opt.metrics # 0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity
    assert metrics == 0 or metrics == 1 or metrics == 2
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Only works if metrics = 0
    EM = 0; precision = 0; recall = 0; fmeasure = 0; # Only works if metrics = 0 | 2
    sem_sim = 0 # Only works if metrics = 2
    Sacc = 0; Lacc = 0 # Only works if metrics = 1
    samples = []
    first_iter = True if opt.forced == 0 else False

    using_batch_api = opt.using_batch_api
    using_prev_file = opt.eval_only
    assert (using_batch_api == 0) or (using_batch_api == 1 and model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct'])
    assert using_prev_file == 0 or (using_prev_file == 1 and using_batch_api == 1)
    
    adaptive_retrieval = opt.adaptive_retrieval
    assert adaptive_retrieval == 0 or (adaptive_retrieval == 1 and top_k > 0)

    filetype = opt.filetype
    assert filetype == 'dev' or filetype == 'test' or filetype == 'train'

    batch_id = opt.batch_id
    if using_prev_file == 1:
        assert batch_id != None

    ##### <using batch api for OpenAI models> #####
    if using_batch_api == 1 and using_prev_file == 0:
        samples = []
        for i in trange(n):
            datum = data[i]
            question = datum[question_key]
            # answers = []
            # datum_2 = answers_data[i]
            # for answer_key in answer_keys:
            #     if type(datum_2[answer_key]) == list:
            #         answers.extend(datum_2[answer_key])
            #     elif type(datum_2[answer_key]) == str:
            #         answers.append(datum_2[answer_key])
            #     else:
            #         assert 1 == 0, 'An answer found that is neither a string nor a list.'
            id = datum[id_key] if id_key != None else i
            try:
                requires_retrieval = datum['requires_retrieval']
            except:
                requires_retrieval = 0
                print('no adaptive retrieval')
            ##### <defining the prompt> #####
            full_prompt = ''; initial_prompt = ''
            if (adaptive_retrieval and requires_retrieval) or ((not adaptive_retrieval) and (top_k > 0)): # include_contexts = 'adaptive'
                if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
                    # initial_prompt = f"Your task is to answer the following question in 2 to 3 words and in a format of factoid answer with respect to the given contexts. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers.\n\nQuestion: {question}"
                    initial_prompt = f"Your task is to answer the following question in 2 to 3 words and in a format of factoid answer with respect to the given contexts. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers.\n\nQuestion: {question}"
                else: # Sacc and Lacc (Strict and Lenient Accuracy)
                    initial_prompt = f"Answer the following question with respect to the given contexts by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
                full_prompt = f"{initial_prompt}\n\nContexts:\n"
                contexts = datum['ctxs']
                full_prompt = add_contexts(full_prompt, contexts, top_k, include_titles)
            else: # include_contexts = False
                if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
                    full_prompt = f"Your task is to answer the following question in 2 to 3 words and in a format of factoid answer. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers.\n\nQuestion: {question}"
                else: # Sacc and Lacc (Strict and Lenient Accuracy)
                    full_prompt = f"Answer the following question by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
            ##### </defining the prompt> #####
            ##### <prompt the user with the estimated total cost before the first iteration> #####
            if first_iter:
                estimated_cost, _ = estimate_cost(n, initial_prompt if top_k > 0 else full_prompt, model, data, max_tokens, list(map(lambda x: list(map(lambda y: y['title'] + '\n' + y['text'], x['ctxs'][:top_k])), data)), using_batch_api=using_batch_api)
                res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
                assert res.strip().lower() == 'y', "User decided to abort the process."
                first_iter = False
            ##### </prompt the user with the estimated total cost before the first iteration> #####
            body = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": full_prompt
                }],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": model_top_p,
                "logprobs": True,
                "top_logprobs": 20,
            }
            samples.append({
                "custom_id": str(id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            })
        batch_file_dir = f"new_results/{dataset}/{model}/batch_files/"
        if not os.path.exists(batch_file_dir):
            os.makedirs(batch_file_dir)
        batch_file_path = f"new_results/{dataset}/{model}/batch_files/{dataset}-{model}-{filetype}-SAG.jsonl"
        with open(batch_file_path, 'w') as file:
            for sample in samples:
                file.write(json.dumps(sample) + '\n')
        client = OpenAI(api_key=api_key)
        batch_input_file = client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        batch_object = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"{dataset}-{model}-{filetype}-SAG"
            }
        )
        print(f"batch_id: {batch_object.id}") # BATCH FILE -|--------------------------------------------------------->>>>
    elif using_batch_api == 1 and using_prev_file == 1:
        samples = []
        client = OpenAI(api_key=opt.api_key)
        output_file_id = client.batches.retrieve(batch_id).output_file_id
        file_response = client.files.content(output_file_id)
        lines = file_response.text.splitlines()
        list_of_dicts = [json.loads(line) for line in lines]
        nlp = spacy.load("en_core_web_md")
        for raw_response, sample_index in zip(list_of_dicts, trange(n)):
            response = raw_response['response']
            id = data[sample_index][id_key] if id_key != None else sample_index
            question = data[sample_index][question_key]
            question_entities = get_entities(question, nlp)
            answers = []
            datum = data[sample_index]
            # assert len(datum[answer_keys[0]]) == 1
            # assert len(datum[answer_keys[0]][0][answer_keys[1]]) == 1
            # answers = datum[answer_keys[0]][0][answer_keys[1]]
            answers = datum[answer_keys[0]]
            # datum_2 = answers_data[sample_index]
            # for answer_key in answer_keys:
            #     if type(datum_2[answer_key]) == list:
            #         answers.extend(datum_2[answer_key])
            #     elif type(datum_2[answer_key]) == str:
            #         answers.append(datum_2[answer_key])
            #     else:
            #         assert 1 == 0, 'An answer found that is neither a string nor a list.'
            generated_answer = response['body']['choices'][0]['message']['content']
            generated_answer_entities = get_entities(generated_answer, nlp)

            token_probabilities, possible_tokens, generated_tokens, generated_tokens_indices, top_p = [], [], [], [], []

            x = 0 # num_answers is set to 1 by default
            if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
                for j in range(len(response['body']['choices'][x]['logprobs']['content'])):
                    top_20_logprobs = response['body']['choices'][x]['logprobs']['content'][j]['top_logprobs']
                    token_probabilities.append(list(map(lambda x: np.exp(x['logprob']), top_20_logprobs)))
                    possible_tokens.append(list(map(lambda x: x['token'], top_20_logprobs)))
                    generated_token = response['body']['choices'][x]['logprobs']['content'][j]['token']
                    generated_tokens.append(generated_token)
                    try:
                        generated_token_index = list(map(lambda x: x['token'], top_20_logprobs)).index(generated_token)
                        generated_tokens_indices.append(generated_token_index)
                        top_p.append(token_probabilities[0][generated_token_index])
                    except:
                        generated_tokens_indices.append(None)
                        top_p.append(None)
            
            entropies = get_token_level_entropies(token_probabilities)
            words, word_entropies = get_word_level_entropies(generated_tokens, entropies)
            word_entropies, dependent_words = combine_dependent_words_entropies(words, deepcopy(word_entropies), generated_answer_entities)

            mets = evaluate_factoid_answers(metrics, answers, generated_answer, scorer)
            EM += mets[0]; precision += mets[1]; recall += mets[2]; fmeasure += mets[3]; sem_sim += mets[4]; Sacc += mets[5]; Lacc += mets[6]

            samples.append({'id': id, 'question': question, 'gold_answers': answers, 'pred_answer': generated_answer, 'exact_match': int(mets[0]), 'word_level_entropies': list(zip(words, list(map(lambda x: round(x, 3), word_entropies)))), 'question_entities': question_entities, 'answer_entities': generated_answer_entities, 'dependent_words': dependent_words, 'word_entropies_median': round(np.percentile(word_entropies, 50), 3)})
            ##### <printing logs> #####
            if v:
                print(f'\nP: {highlight_ngrams(question, color="blue")}')
                color = ''
                if metrics == 0:
                    color = 'green' if mets[0] else 'red'
                else:
                    color = 'green' if mets[6] else 'red'
                print(f'A: {answers}\nO: {highlight_ngrams(generated_answer, color=color)}')
                if metrics == 0:
                    print(f'EM: {mets[0]:.3f}')
                elif metrics == 1:
                    print(f'Lacc: {mets[6]:.3f}')
                elif metrics == 2:
                    print(f'Sem-Sim: {mets[4]:.3f}')
            ##### </printing logs> #####
        ##### <printing evaluation results> #####
        if metrics == 0 or metrics == 2:
            acc = round(EM/n, 3); pre = round(precision/n, 3); rec = round(recall/n, 3); f1 = round(fmeasure/n, 3)
            print(f'\nEM: {EM}')
            print(f"Accuracy: {acc}")
            print(f"Precision: {pre}")
            print(f"Recall: {rec}")
            print(f"F-1: {f1}")
        elif metrics == 1:
            Sa = round(Sacc/n, 3); La = round(Lacc/n, 3)
            print(f'\nSacc: {Sa}')
            print(f"Lacc: {La}")
        if metrics == 2:
            ss = round(sem_sim/n, 3)
            print(f"Sem-Sim: {ss}")
        ##### </printing evaluation results> #####
        ##### <saving the results> #####
        # if template_temperature == 'high':
        #     file_path = f'new_results/{dataset}/{model}/{retriever}/{dataset}_{filetype}_1.0_1.0_{percentile}_{model}_{LAG}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
        # else:
        #     file_path = f'new_results/{dataset}/{model}/{retriever}/{dataset}_{filetype}_{percentile}_{model}_{LAG}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
        file_path = f'final_results/{dataset}/{model}/{filetype}_short_T{temperature}_AR{adaptive_retrieval}_k{top_k}_{query_type}-M3A.jsonl'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        ##### </saving the results> #####
    ##### </using batch api for OpenAI models> ####
    if using_batch_api == 0:
        for i in trange(n):
            datum = data[i]
            question = datum[question_key]
            answers = []
            # assert len(datum[answer_keys[0]]) == 1
            # assert len(datum[answer_keys[0]][0][answer_keys[1]]) == 1
            # answers = datum[answer_keys[0]][0][answer_keys[1]]
            answers = datum[answer_keys[0]]
            # datum_2 = answers_data[i]
            # for answer_key in answer_keys:
            #     if type(datum_2[answer_key]) == list:
            #         answers.extend(datum_2[answer_key])
            #     elif type(datum_2[answer_key]) == str:
            #         answers.append(datum_2[answer_key])
            #     else:
            #         assert 1 == 0, 'An answer found that is neither a string nor a list.'
            id = datum[id_key] if id_key != None else i
            try:
                requires_retrieval = datum['requires_retrieval']
            except:
                requires_retrieval = 0
            ##### <defining the prompt> #####
            full_prompt = ''; initial_prompt = ''
            contexts = datum['ctxs']
            doc_content = add_contexts("", contexts, top_k=top_k, include_titles=include_titles)

            if (adaptive_retrieval and requires_retrieval) or ((not adaptive_retrieval) and (top_k > 0)): # include_contexts = 'adaptive'
                system = "System: This is a chat between a user and an artificial intelligence assistant."
                if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity

                    #### No C-O-T  for GPT models####
                    initial_prompt = (
                    
                    # "You are a helpful assistant skilled in answering True/False Multi-hop questions given the provided knowledge, your task is to answer the given question after <Question>.\n"
                    # "You must refer to this knowledge when answering the question.\n"
                    # "The answer must be concise ***True or False***, nothing more.\n"
                    # "If the knowledge does not contain the answer, **do not generate anything**.\n"
                    

                    # "Here are some examples:\n\n"

                    # "Example 1:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
                    # "So, the answer is: False\n\n"

                    # "Example 2:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
                    # "So, the answer is: True\n\n"

                    # "Example 3:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
                    # "So, the answer is: True\n\n"
        

                    "You are an expert assistant in answering complex and **Multi-Hop** questions after <Question>. "
                    "Your task is to provide a concise, factoid-style answer based strictly on the information enclosed within <doc> and </doc> tags.\n\n"

                    "**Instructions:**\n"
                    "1. Use logical reasoning *only* if the document lacks direct information.\n"
                    "2. DO NOT include your thought process or explanation in the final output.\n\n"

                    "**Formatting Rules:**\n"
                    "- If the question is yes/no, answer with **yes** or **no** only.\n"
                    "- If the answer is a date or place, give only the date or place.\n"
                    "- If the answer is not clearly stated, use reasoning but respond in a maximum of 2–3 words.\n"
                    "- If you DO NOT know the answer, DO NOT generate anything.\n\n"

                    "**Examples:**\n\n"

                    "Example 1:\n"
                    "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
                    "So, the answer is: Little Richard\n"

                    "Example 2:\n"
                    "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
                    "So, the answer is: Chinua Achebe\n"

                    "Example 3:\n"
                    "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
                    "So, the answer is: 1979\n\n"

                    )

                    #### C-O-T for GPT Models####
                    # initial_prompt = (

                    # "You are a helpful assistant skilled in answering **True/False Multi-hop** questions based on the provided knowledge.\n "
                    # "Your task is to answer the given question (enclosed within <Question>) using only the information inside <doc> tags.\n"
                    # "You **must** think through the information **step by step**, providing your internal reasoning before answering.\n"
                    # "If the provided knowledge contains relevant information, use it to generate a logical, multi-step thought process.\n "
                    # "If the knowledge does not contain an explicit answer, reason independently and provide your best judgment.\n"
                    # "Your final response must be a single, concise answer: either ***True*** or ***False*** — nothing else.\n"
                    # "**IMPORTANT:** Your output must ONLY contain the final answer. Do NOT include your thoughts, explanations, or any prefixes in the final output.\n\n"


                    # "Here are some examples:\n\n"

                    # "Example 1:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Laxatives are substances that loosen stools and increase bowel movements. People take laxatives to treat and/or prevent constipation.\n"
                    # "So, the answer is: False\n\n"

                    # "Example 2:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Durian has a pungent odor that many people describe as being similar to feet and onions. Unpleasant smells can make people feel nauseous.\n"
                    # "So, the answer is: True\n\n"


                    # "Example 3:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Monty Python and the Holy Grail was a famous film about King Arthur. In Monty Python and the Holy Grail, swallows are mentioned several times.\n"
                    # "So, the answer is: True\n\n"
                    
                
                    # "Let’s think step by step:"
                            
                    # "You are an expert assistant in answering complex and **Multi-Hop** questions. "
                    # "Your task is to provide a concise, factoid-style answer based strictly on the information enclosed within <doc> and </doc> tags.\n\n"

                    # "**Instructions:**\n"
                    # "1. Think step-by-step using the document content.\n"
                    # "2. Use logical reasoning *only* if the document lacks direct information.\n"
                    # "3. DO NOT include your thought process or explanation in the final output.\n"


                    # "**Formatting Rules:**\n"
                    # "- If the question is yes/no, answer with **yes** or **no** only.\n"
                    # "- If the answer is a date or place, give only the date or place.\n"
                    # "- If the answer is not clearly stated, use reasoning but respond in a maximum of 2–3 words.\n"
                    # "- If you DO NOT know the answer, DO NOT generate anything.\n\n"

                    # "**Examples:**\n\n"

                    # "Example 1:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
                    # # "<Question>: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?\n"
                    # "Let's think step-by-step:\n"
                    # "<Thought>: Modern Records was an R&B label with artists including Etta James, Joe Houston, Little Richard, Ike & Tina Turner, and John Lee Hooker in the 1950s and 1960s. The given knowledge confirms that Little Richard was an American musician, singer, actor, and songwriter born on December 5, 1932, and worked with Modern Records.\n"
                    # # "<Thought>: Valentin The Good is directed by Martin Fric. Martin Fri ˇ c was a Czech film director. ˇ The Falcon (Film) is directed by Vatroslav Mimica. Vatroslav Mimica is a Croatian film director. Czech is different from Croatia.\n"
                    # "So, the answer is: Little Richard\n"
                    # # "So, the answer is: no\n"


                    # "Example 2:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
                    # # "<Question>: What nationality is the director of film Wedding Night In Paradise (1950 Film)?\n"
                    # "Let's think step-by-step:\n"
                    # "<Thought>: Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Since Chinua Achebe had four different jobs while Rachel Carson had three, the answer is Chinua Achebe.\n"
                    # # "<Thought>: Wedding Night In Paradise (1950 film) is directed by Géza von Bolváry. Géza von Bolváry was a Hungarian actor, screenwriter and film director.\n"
                    # "So, the answer is: Chinua Achebe\n"
                    # # "So, the answer is: Hungarian\n"

                    # "Example 3:\n"
                    # "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                    # "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
                    # # "<Question>: Who is Rhescuporis I (Odrysian)’s paternal grandfather?\n"
                    # "Let's think step-by-step:\n"
                    # "<Thought>: Remember Me Ballin’ is a CD single by Indo G that features Gangsta Boo. The given knowledge states that Gangsta Boo, whose real name is Lola Mitchell, is an American rapper born in 1979.\n"
                    # # "<Thought>: The father of Rhescuporis I (Odrysian) is Cotys III. The father of Cotys III is Raizdos.\n"
                    # "So, the answer is: 1979\n\n"
                    # # "So, the answer is: Raizdos\n\n"
                    
                    
                    # "Let's think step-by-step:"

                    # )

                    full_prompt = (
                        f"{system}\n\n"
                        f"User:{initial_prompt}\n"
                        f"Now use the given knowledge below to answer the question. Internally reason step-by-step, but Output only the final answer, nothing else.\n\n" #Internally reason step-by-step, but 
                        f"<doc>\n{doc_content}\n</doc>\n"
                        f"<Question>: {datum['question_org']}\nAssistant:\nSo, the answer is:"
                        # f"Let’s think step by step:\n"
                        )

                    print(f"Full Prompt is:\n{full_prompt}")

                else: # Sacc and Lacc (Strict and Lenient Accuracy)
                    initial_prompt = f"Answer the following question with respect to the given contexts by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
                    full_prompt = f"{initial_prompt}\n\nContexts:\n" 
                    contexts = datum['ctxs']
                    full_prompt = add_contexts(full_prompt, contexts, top_k, include_titles)


            else: # include_contexts = False
                if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
                    
                    #No C-O-T#
                    initial_prompt = (
                        
                    # "You are a helpful assistant skilled in answering True/False Multi-hop questions, your task is to answer the given question after <Question>.\n"
                    # "You must use logical reasoning to arrive at the best possible answer\n"
                    # "The answer must be concise **True** or **False***, nothing more.\n"
                    # "Here are some examples:\n\n"

                    # "Example 1:\n"
                    # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
                    # "So, the answer is: False\n\n"

                    # "Example 2:\n"
                    # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
                    # "So, the answer is: True\n\n"

                    # "Example 3:\n"
                    # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
                    # "So, the answer is: True\n\n"
                        
                    "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>."
                    "You must use logical reasoning to arrive at the best possible answer.\n"
                    "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
                    "In case of referring to a date or specific place, just name the date or place.\n"
                    "The answer must be concise (2–3 words max), factoid-style.\n"  


                    "Here are some examples:\n\n"


                    "Example 1:\n"
                    "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
                    "So, the answer is: Little Richard\n\n"

                    "Example 2:\n"
                    "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
                    "So, the answer is: Chinua Achebe\n\n"

                    "Example 3:\n"
                    "<Question>: 'Remember Me Ballin’' is a CD single by Indo G that features an American rapper born in what year?\n"
                    "So, the answer is: 1979\n\n"
                    )


                    ##C-O-T#
                    # initial_prompt = (

                    # "You are a helpful assistant skilled in answering True/False Multi-hop questions, your task is to answer the given question after <Question>.\n"
                    # "You Must First Think about the question **step-by-step**, give your thoughts, and then answer the question after <Question>.\n"
                    # "The answer must be concise ***True or False***, nothing more.\n"
                    # "**If you DO NOT know the answer, DO NOT generate anything.**\n"
                    # "IMPORTANT: Output ONLY the final answer. Do NOT include the **thought** process or any other prefixes in your final response.\n\n"
                    # "Here are some examples:\n\n"

                    # "Example 1:\n"
                    # "<Question>: Do people take laxatives because they enjoy diarrhea?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Laxatives are substances that loosen stools and increase bowel movements. People take laxatives to treat and/or prevent constipation.\n"
                    # "So, the answer is: False\n\n"

                    # "Example 2:\n"
                    # "<Question>: Could Durian cause someone’s stomach to feel unwell?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Durian has a pungent odor that many people describe as being similar to feet and onions. Unpleasant smells can make people feel nauseous.\n"
                    # "So, the answer is: True\n\n"

                    # "Example 3:\n"
                    # "<Question>: Did the swallow play a role in a famous film about King Arthur?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Monty Python and the Holy Grail was a famous film about King Arthur. In Monty Python and the Holy Grail, swallows are mentioned several times.\n"
                    # "So, the answer is: True\n\n"
                    
                    # "Let’s think step by step:"
                    
                    # "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>.\n"
                    # "You Must First Think about the question **step-by-step**, give your thoughts, and then answer the question after <Question>.\n"
                    # "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
                    # "In case of referring to a date or specific place, just name the date or place.\n"
                    # "The answer must be concise (2–3 words max), factoid-style, and based strictly on the provided knowledge.\n" 
                    # "**If you DO NOT know the answer, DO NOT generate anything.**\n"
                    # "IMPORTANT: Output ONLY the final answer. Do NOT include the **thought** process or any other prefixes in your final response.\n\n"
                
                    # "Here are some examples:\n\n"
                    # "Example 1:\n"
                    # "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Modern Records was an R&B label with artists including Etta James, Joe Houston, Little Richard, Ike & Tina Turner, and John Lee Hooker in the 1950s and 1960s. The given knowledge confirms that Little Richard was an American musician, singer, actor, and songwriter born on December 5, 1932, and worked with Modern Records\n"
                    # "So, the answer is: Little Richard\n\n"

                    # "Example 2:\n"
                    # "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Since Chinua Achebe had four different jobs while Rachel Carson had three, the answer is Chinua Achebe.\n"
                    # "So, the answer is: Chinua Achebe\n\n"

                    # "Example 3:\n"
                    # "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
                    # "Let’s think step by step.\n"
                    # "<Thought>: Remember Me Ballin’ is a CD single by Indo G that features Gangsta Boo. The given knowledge states that Gangsta Boo, whose real name is Lola Mitchell, is an American rapper born in 1979.\n"
                    # "So, the answer is: 1979\n\n"

                    # "Let’s think step by step:"
                    
                    # )

                    full_prompt = (
                    f"System: This is a chat between a user and an artificial intelligence assistant\n\nUser:{initial_prompt}\n<Question>:{datum['question_org']}\n\nAssistant:\nSo, the answer is:")
                    print(f"the full prompt w/o contexts is: {full_prompt}")

                else: # Sacc and Lacc (Strict and Lenient Accuracy)
                    full_prompt = f"Answer the following question by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
            
            ##### </defining the prompt> #####
            ##### <prompt the user with the estimated total cost before the first iteration> #####
            if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct'] and first_iter:
                estimated_cost, _ = estimate_cost(n, initial_prompt if top_k > 0 else full_prompt, model, data, max_tokens, list(map(lambda x: list(map(lambda y: y['title'] + '\n' + y['text'], x['ctxs'][:top_k])), data)), using_batch_api=using_batch_api)
                res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
                assert res.strip().lower() == 'y', "User decided to abort the process."
                first_iter = False
            ##### </prompt the user with the estimated total cost before the first iteration> #####
            if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct']:
                generated_answer = openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p) 
                # temperatures = [float(t) for t in args.temperature.split(',')]                    
                # generated_answers = []
                # for temp in temperatures:
                #     generated_answers.append(openai_generator(api_key, model, full_prompt, max_tokens, temp, model_top_p))

                # from collections import Counter
                # answer_counts = Counter(generated_answers)
                # generated_answer, _ = answer_counts.most_common(1)[0]
                
                print(f"Generated Answer for ID {id}: {generated_answer}")
                print(f"\nPrompt used for ID {id}:\n{full_prompt}\n")

            elif model in ['llama-3', 'llama-3.1', 'qwen', 'vicuna']:
                if (adaptive_retrieval and requires_retrieval) or ((not adaptive_retrieval) and (top_k > 0)):
                    # temperatures = [float(t) for t in args.temperature.split(',')]                    
                    # generated_answers = []
                    # for temp in temperatures:
                    #     generated_answers.append(meta_generator(meta_model, meta_tokenizer, question, max_tokens, temp, model_top_p, top_k, contexts))

                    # from collections import Counter
                    # answer_counts = Counter(generated_answers)
                    # generated_answer, _ = answer_counts.most_common(1)[0]

                    generated_answer = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, top_k, contexts)
                    print(f"Generated Answer for ID {id}: {generated_answer}")
                    
                else:
                    generated_answer = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, top_k)
                    print(f"Generated Answer for ID {id}: {generated_answer}")
                #     print(f"\nPrompt used for ID {id}:\n{full_prompt}\n")
            mets = evaluate_factoid_answers(metrics, answers, generated_answer, scorer)
            EM += mets[0]; precision += mets[1]; recall += mets[2]; fmeasure += mets[3]; sem_sim += mets[4]; Sacc += mets[5]; Lacc += mets[6]

            samples.append({'id': id, 'question': question, 'gold_answers': answers, 'pred_answer': generated_answer, 'exact_match': int(mets[0])})
            ##### <printing logs> #####
            if v:
                print(f'\nP: {highlight_ngrams(full_prompt, color="blue")}')
                color = ''
                if metrics == 0:
                    color = 'green' if mets[0] else 'red'
                else:
                    color = 'green' if mets[6] else 'red'
                print(f'A: {answers}\nO: {highlight_ngrams(generated_answer, color=color)}')
                if metrics == 0:
                    print(f'EM: {mets[0]:.3f}')
                elif metrics == 1:
                    print(f'Lacc: {mets[6]:.3f}')
                elif metrics == 2:
                    print(f'Sem-Sim: {mets[4]:.3f}')
            ##### </printing logs> #####
        ##### <printing evaluation results> #####
        if metrics == 0 or metrics == 2:
            acc = round(EM/n, 3); pre = round(precision/n, 3); rec = round(recall/n, 3); f1 = round(fmeasure/n, 3)
            print(f'\nEM: {EM}')
            print(f"Accuracy: {acc}")
            print(f"Precision: {pre}")
            print(f"Recall: {rec}")
            print(f"F-1: {f1}")
        elif metrics == 1:
            Sa = round(Sacc/n, 3); La = round(Lacc/n, 3)
            print(f'\nSacc: {Sa}')
            print(f"Lacc: {La}")
        if metrics == 2:
            ss = round(sem_sim/n, 3)
            print(f"Sem-Sim: {ss}")
        ##### </printing evaluation results> #####
        ##### <saving the results> #####
        # if template_temperature == 'high':
        #     file_path = f'new_results/{dataset}/{model}/{retriever}/{dataset}_{filetype}_1.0_1.0_{percentile}_{model}_{LAG}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
        # else:
        #     file_path = f'new_results/{dataset}/{model}/{retriever}/{dataset}_{filetype}_{percentile}_{model}_{LAG}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
        file_path = f'final_results/{dataset}/{model}/{filetype}_short_T{temperature}_AR{adaptive_retrieval}_k{top_k}_{query_type}.jsonl'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        ##### </saving the results> #####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help="Path to the data")
    parser.add_argument('--data_org', required=False, type=str, default=None, 
                        help="Path to the original data")
    parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Model name: gpt-3.5-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo-instruct, llama-3 (llama3-ChatQA-1.5-8B), llama-3.1 (Llama-3.1-8B)")
    parser.add_argument('--top_k', required=False, type=int, default=0, 
                        help="Top-k contexts used for Retrieval-Augmented Generation (RAG)")
    parser.add_argument('--include_titles', required=False, type=int, default=1, 
                        help="0: Do not include the titles for the contexts | 1: Include the titles for the contexts | Only works when top_k > 0")
    parser.add_argument('--api_key', required=False, type=str, default=None, 
                        help="OpenAI's API key")
    parser.add_argument('--max_tokens', required=False, type=int, default=50, 
                        help="Maximum number of the output tokens")
    # parser.add_argument('--temperature', required=False, type=str, default="0.0,0.7,0.7,0.7,0.7", 
    #                                     help="Comma-separated list of temperatures")                    
    parser.add_argument('--temperature', required=False, type=float, default=0, 
                        help="Model config")
    parser.add_argument('--top_p', required=False, type=float, default=1, 
                        help="Model config")
    parser.add_argument('--metrics', required=False, type=int, default=0, 
                        help="0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity")
    parser.add_argument('--eval_only', required=False, type=int, default=0, 
                        help="0: Generate factoid answers, then evaluate | 1: Only evaluate a pre-saved file")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")
    # parser.add_argument('--LAG', required=False, type=str, default='gpt-3.5-turbo', 
    #                     help="Long-form Answer Generator (LAG): gpt-3.5-turbo, gpt-4o, llama-3, gpt-4o-mini, gpt-3.5-turbo-instruct")
    parser.add_argument('--forced', required=False, type=int, default=0, 
                        help="Skip the total cost estimation process for OpenAI models: 1")
    parser.add_argument('--using_batch_api', required=False, type=int, default=0, 
                        help="Set to 0 if not using the Batch API")
    parser.add_argument('--batch_id', required=False, type=str, default=None, 
                        help="Batch ID")
    parser.add_argument('--adaptive_retrieval', required=False, type=int, default=0, 
                        help="Set to 0 for non-adaptive retrieval")
    parser.add_argument('--filetype', required=False, type=str, default='test', 
                        help="test | dev")
    parser.add_argument('--dataset', required=True, type=str, default=None, 
                        help="MuSiQue, IIRC, HotpotQA, 2WikiMultihopQA, StrategyQA, MuSiQue")
    # parser.add_argument('--answers', required=False, type=str, default=None, 
    #                     help="Answers file (main dataset)")
    parser.add_argument('--query_type', required=False, type=str, default=None, 
                        help="question, long_T0, long_mixed, and long_mixed_classified")

    args = parser.parse_args()
    
    if args.eval_only and args.using_batch_api == 0:
        evaluate_saved_file(args)
    else:
        main(args)