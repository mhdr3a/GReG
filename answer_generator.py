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

def estimate_cost(total_samples, fixed_instructions_msg, model, questions, max_output_tokens_per_sample, contexts=None, word_to_token_ratio=1.33, include_contexts=True, question_key='question'):
    fixed_instructions_tokens = word_to_token_ratio * len(fixed_instructions_msg.replace('\n', ' ').strip().split())
    total_context_tokens = word_to_token_ratio * sum(list(map(lambda x: sum(list(map(lambda y: len(y.replace('\n', ' ').strip().split()), x))), contexts))) if include_contexts else 0
    total_input_tokens = word_to_token_ratio * sum(list(map(lambda q: len(q[question_key].replace('\n', ' ').strip().split()), questions))) + total_samples * fixed_instructions_tokens + total_context_tokens
    total_output_tokens = total_samples * max_output_tokens_per_sample
    cost_per_1M_tokens_input = (
        5.0   if model == 'gpt-4o' else
        0.60  if model == 'gpt-4o-mini' else
        10.0  if model == 'gpt-4-turbo' else
        0.50  if model == 'gpt-3.5-turbo' else
        None
    )

    cost_per_1M_tokens_output = (
        20.0  if model == 'gpt-4o' else
        2.40  if model == 'gpt-4o-mini' else
        30.0  if model == 'gpt-4-turbo' else
        1.50  if model == 'gpt-3.5-turbo' else
        None
    )

    total_tokens, estimated_cost = total_input_tokens + total_output_tokens, cost_per_1M_tokens_input * (total_input_tokens / 1000000) + cost_per_1M_tokens_output * (total_output_tokens / 1000000)

    return estimated_cost, total_tokens

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

def openai_generator_ins(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
    import openai
    openai.api_key = api_key

    response = openai.completions.create(
        model=model,
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=model_top_p,
    )
    return response.choices[0].text.strip()

import time
import openai
from openai import RateLimitError

def openai_generator_non_ins(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
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

def openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
    if "instruct" in model:
        return openai_generator_ins(api_key, model, full_prompt, max_tokens, temperature, model_top_p)
    else:
        return openai_generator_non_ins(api_key, model, full_prompt, max_tokens, temperature, model_top_p)


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

    return decoded

def meta_get_formatted_input(question, top_k, contexts, include_titles=True):
    
    system = "System: This is a chat between a user and an artificial intelligence assistant."
    if contexts is None:
        instruction = (
            "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>."
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

        conversation = f"User:{instruction}\n<Question>: {question}\n\n"
        formatted_input = f"{system}\n\n{conversation}Assistant:\nSo, the answer is:"

        print(formatted_input)
    else:
        instruction = (
            "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question> based on the provided knowledge enclosed within <doc> and </doc> tags.\n"
            "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
            "In case of referring to a date or specific place, just name the date or place.\n"
            "If the knowledge contains the answer, give a **concise, factoid-style answer (2–3 words max)**.\n"
            "**IMPORTANT: Do NOT include any thought process, explanation, or reasoning. Only return the final answer after <Answer>.**\n"

            "Here are some examples:\n\n"

            "Example 1:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
            "So, the answer is: Little Richard\n\n"

            "Example 2:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
            "So, the answer is: Chinua Achebe\n\n"

            "Example 3:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
            "So, the answer is: 1979\n\n"
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
    dataset = opt.dataset
    assert dataset in ['MuSiQue', 'HotpotQA', '2WikiMultihopQA']
    
    data, id_key, n = read_data(filepath, ext)

    top_k = opt.top_k
    assert 0 <= top_k <= 10

    answer_keys = ['answer']
    
    model = opt.model
    assert model in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct', 'vicuna-13b', 'qwen-7b']
    if model == 'llama-3':
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

    max_tokens = opt.max_tokens
    assert max_tokens > 0

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

    filetype = opt.filetype
    assert filetype == 'dev' or filetype == 'test' or filetype == 'train'

    question_key = 'question'

    for i in trange(n):
        datum = data[i]
        question = datum[question_key]
        answers = []

        answers = datum[answer_keys[0]]

        id = datum[id_key] if id_key != None else i

        ##### <defining the prompt> #####
        full_prompt = ''; initial_prompt = ''
        contexts = datum['ctxs']
        doc_content = add_contexts("", contexts, top_k=top_k, include_titles=include_titles)

        if top_k > 0:
            system = "System: This is a chat between a user and an artificial intelligence assistant."
            if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity

                #### GPT Models ####
                initial_prompt = (
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

                full_prompt = (
                    f"{system}\n\n"
                    f"User:{initial_prompt}\n"
                    f"Now use the given knowledge below to answer the question. Internally reason step-by-step, but Output only the final answer, nothing else.\n\n" 
                    f"<doc>\n{doc_content}\n</doc>\n"
                    f"<Question>: {datum['question_org']}\nAssistant:\nSo, the answer is:"
                )

                print(f"Full Prompt is:\n{full_prompt}")

            else: # Sacc and Lacc (Strict and Lenient Accuracy)
                initial_prompt = f"Answer the following question with respect to the given contexts by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
                full_prompt = f"{initial_prompt}\n\nContexts:\n" 
                contexts = datum['ctxs']
                full_prompt = add_contexts(full_prompt, contexts, top_k, include_titles)


        else: # top_k = 0
            if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
                
                # GPT Models #
                initial_prompt = (   
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

                full_prompt = (
                f"System: This is a chat between a user and an artificial intelligence assistant\n\nUser:{initial_prompt}\n<Question>:{datum['question_org']}\n\nAssistant:\nSo, the answer is:")
                print(f"the full prompt w/o contexts is: {full_prompt}")

            else: # Sacc and Lacc (Strict and Lenient Accuracy)
                full_prompt = f"Answer the following question by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
        
        ##### </defining the prompt> #####
        ##### <prompt the user with the estimated total cost before the first iteration> #####
        if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct']:
            estimated_cost, _ = estimate_cost(n, initial_prompt if top_k > 0 else full_prompt, model, data, max_tokens, list(map(lambda x: list(map(lambda y: y['title'] + '\n' + y['text'], x['ctxs'][:top_k])), data)))
            res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
            assert res.strip().lower() == 'y', "User decided to abort the process."
        ##### </prompt the user with the estimated total cost before the first iteration> #####
        if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct']:
            generated_answer = openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p) 
            
            print(f"Generated Answer for ID {id}: {generated_answer}")
            print(f"\nPrompt used for ID {id}:\n{full_prompt}\n")

        elif model in ['llama-3', 'qwen-7b', 'vicuna-13b']:
            if top_k > 0:
                generated_answer = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, top_k, contexts)
                print(f"Generated Answer for ID {id}: {generated_answer}")
            else:
                generated_answer = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, top_k)
                print(f"Generated Answer for ID {id}: {generated_answer}")
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
    file_path = f'answers/{dataset}/{model}/{filetype}_short_T{temperature}_k{top_k}.jsonl'
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
    parser.add_argument('--temperature', required=False, type=float, default=0, 
                        help="Model config")
    parser.add_argument('--top_p', required=False, type=float, default=1, 
                        help="Model config")
    parser.add_argument('--metrics', required=False, type=int, default=0, 
                        help="0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")
    parser.add_argument('--filetype', required=False, type=str, default='test', 
                        help="test | dev")
    parser.add_argument('--dataset', required=True, type=str, default=None, 
                        help="MuSiQue, HotpotQA, 2WikiMultihopQA")

    args = parser.parse_args()
    
    main(args)
