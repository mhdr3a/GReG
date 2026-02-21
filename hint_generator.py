import argparse
import openai
from openai import OpenAI
import numpy as np
# from IPython.display import display, HTML
from time import time
import re
import spacy
from tqdm import trange
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax
from copy import deepcopy
from termcolor import colored
import math
import json
import os
import string
from unidecode import unidecode

BATCH_FILE_LENGTH_LIMIT = 50000

# def map_entropy_to_color(entropy, min_entropy, max_entropy, inv):
#     if max_entropy != min_entropy:
#         normalized_entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
#     else:
#         normalized_entropy = entropy
#     colormap = plt.get_cmap('coolwarm') if inv==False else plt.get_cmap('coolwarm_r')
#     rgba_color = colormap(normalized_entropy)
#     hex_color = mcolors.to_hex(rgba_color)
#     return hex_color

# def colorize_tokens(tokens, entropies, threshold=None, inv=False, mask=True):
#     min_entropy = min(entropies)
#     max_entropy = max(entropies)
#     colorized_tokens = []
#     for token, entropy in zip(tokens, entropies):
#         color = map_entropy_to_color(entropy, min_entropy, max_entropy, inv)
#         if threshold == None or entropy < threshold:
#             colorized_tokens.append(f'<span style="color:{color}">{token}</span>')
#         else:
#             if mask == False:
#                 colorized_tokens.append(f'<span style="color:{color}">{token}</span>')
#             else:
#                 colorized_tokens.append(f'<span style="color:{color}">_</span>')
#     return ' '.join(colorized_tokens)

def normalize(str1):
    str1 = str(str1)
    str1_normalized = re.sub(r'[^\w\d]', '', unidecode(re.sub(r'\b(a|an|the)\b', '', str1, flags=re.IGNORECASE)).lower())
    return str1_normalized

def highlight_ngrams(sentence, ngrams_to_highlight=[], color='red', highlight_all=True):
    if highlight_all:
        return colored(sentence, color)
    ngrams_to_highlight = sorted(ngrams_to_highlight, key=lambda x: len(x.split()), reverse=True)
    for ngram in ngrams_to_highlight:
        sentence = sentence.replace(ngram, colored(ngram, color))
    return sentence

def estimate_cost(total_samples, fixed_instructions_msg, model, questions, max_output_tokens_per_sample, contexts=None, word_to_token_ratio=1.33, include_contexts=True, question_key='question', using_batch_api=True):
    fixed_instructions_tokens = word_to_token_ratio * len(fixed_instructions_msg.replace('\n', ' ').strip().split())
    total_context_tokens = word_to_token_ratio * sum(list(map(lambda x: sum(list(map(lambda y: len(y.replace('\n', ' ').strip().split()), x))), contexts))) if include_contexts else 0
    total_input_tokens = word_to_token_ratio * sum(list(map(lambda q: len(q[question_key].replace('\n', ' ').strip().split()), questions))) + total_samples * fixed_instructions_tokens + total_context_tokens
    total_output_tokens = total_samples * max_output_tokens_per_sample
    cost_per_1M_tokens_input = 2.5 if model == 'gpt-4o' else (.5 if model == 'gpt-3.5-turbo' else (10 if model == 'gpt-4-turbo' else (.150 if model == 'gpt-4o-mini' else (1.5 if model == 'gpt-3.5-turbo-instruct' else None))))
    cost_per_1M_tokens_output = 10 if model == 'gpt-4o' else (1.5 if model == 'gpt-3.5-turbo' else (30 if model == 'gpt-4-turbo' else (.600 if model == 'gpt-4o-mini' else (2.0 if model == 'gpt-3.5-turbo-instruct' else None))))

    total_tokens, estimated_cost = total_input_tokens + total_output_tokens, cost_per_1M_tokens_input * (total_input_tokens / 1000000) + cost_per_1M_tokens_output * (total_output_tokens / 1000000)

    return (.5 if using_batch_api else 1) * estimated_cost, total_tokens

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

def filter_words(words, word_entropies, dependent_words, question_entities, thresholds, mask=' '):
    all_filtered_words = {}
    question_entities = list(map(normalize, question_entities))
    for threshold in thresholds:
        th = np.percentile(word_entropies, threshold) # threshold = 0 -> ignore_entropies = True, threshold = 100 -> template_query = answer
        filtered_words = []
        for i, (word, entropy) in enumerate(zip(words, word_entropies)):
            if (entropy > th) and (len(dependent_words[i]) > 0) and (not any(normalize(dependent_word) in question_entities for dependent_word in dependent_words[i])):
                filtered_words.append(mask)
                continue
            filtered_words.append(word)
        all_filtered_words[threshold] = filtered_words
    return all_filtered_words

def get_entities(sentence, nlp):
    entities = []
    doc = nlp(sentence)
    for ent in doc.ents:
        entities.append(ent.text)
    entities = list(set(entities))
    return list(map(remove_punctuation, entities))

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

class InvalidFileExtensionError(Exception):
    def __init__(self, extension, message="Invalid file extension"):
        self.extension = extension
        self.message = f"{message}: {extension}"
        super().__init__(self.message)

def meta_get_formatted_input(question, system_instruction):
    system = "System: This is a chat between a user and an artificial intelligence assistant."
    instruction = f"Please give a complete and concise answer to the following question. {system_instruction}"

    conversation = "User: " + instruction + " " + question + "\n\nAssistant:"
    formatted_input = system + "\n\n" + conversation

    return formatted_input

def meta_generator(model, tokenizer, question, max_tokens, temperature, top_p, system_instruction, template_query_generation=False):
    full_prompt = meta_get_formatted_input(question, system_instruction) # make this consistent with the factoid_answer_generator.py file
    tokenized_prompt = tokenizer(tokenizer.bos_token + full_prompt, return_tensors="pt").to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if temperature == 0: # deterministic
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=max_tokens, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id, output_scores=template_query_generation, return_dict_in_generate=template_query_generation)
    else: # stochastic
        assert temperature > 0
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=max_tokens, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=temperature, top_p=top_p, output_scores=template_query_generation, return_dict_in_generate=template_query_generation)
    if template_query_generation == True:
        response = outputs[0][0][tokenized_prompt.input_ids.shape[-1]:]
    else:
        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    if template_query_generation == True:
        top_probs_and_tokens = []
        for scores in outputs.scores:
            probs = softmax(scores, dim=-1)
            top_probs, top_indices = torch.topk(probs, 20) # 20 is the max value for Open AI models

            step_result = {
                'top_probs': [
                    {'token': tokenizer.decode([idx], skip_special_tokens=True), 'prob': prob.item()}
                    for idx, prob in zip(top_indices[0], top_probs[0])
                ],
                'token': tokenizer.decode([torch.argmax(scores[0])], skip_special_tokens=True)
            }
            top_probs_and_tokens.append(step_result)
        return tokenizer.decode(response, skip_special_tokens=True), top_probs_and_tokens
    return tokenizer.decode(response, skip_special_tokens=True), []

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

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    # dataset = filepath.split('/')[-2]
    dataset = opt.dataset
    assert dataset in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA', 'StrategyQA', 'NQ', 'TQA']
    
    data, id_key, n = read_data(filepath, ext)
    
    question_key = 'question'
    answer_keys = ['answer']
    
    model = opt.model
    assert model in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct']
    if model == 'llama-3':
        model_id = "nvidia/Llama3-ChatQA-1.5-8B"
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        meta_tokenizer = AutoTokenizer.from_pretrained(model_id)
    v = opt.verbose
    assert v == 0 or v == 1
    max_tokens = opt.max_tokens
    assert max_tokens > 0
    percentile = list(set(opt.percentile))
    # assert len(percentile) > 0
    k_fold = opt.k_fold
    temperature = opt.temperature
    assert 0 <= temperature <= 2
    assert temperature == 0 or k_fold != None
    model_top_p = opt.top_p
    assert .1 <= model_top_p <= 1

    using_batch_api = opt.using_batch_api
    using_prev_file = opt.using_prev_file
    assert (using_batch_api == 0 and using_prev_file == 0) or (using_batch_api == 1 and model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct'])
    assert using_prev_file == 0 or (using_prev_file == 1 and using_batch_api == 1)

    filetype = opt.filetype
    assert filetype == 'train' or filetype == 'test' or filetype == 'dev'

    batch_id = opt.batch_id
    if using_prev_file == 1:
        assert batch_id != None
    
    segment = opt.segment
    if using_batch_api == 1 and n > BATCH_FILE_LENGTH_LIMIT:
        assert segment != None, f"When using the Batch API, if the input file length is > {BATCH_FILE_LENGTH_LIMIT} you have to provide the segment argument to partition the input file."
    
    if segment != None:
        min_index = segment * BATCH_FILE_LENGTH_LIMIT
        max_index = min(n - min_index, BATCH_FILE_LENGTH_LIMIT) + min_index
    else:
        min_index = 0; max_index = n
    
    system_instruction = "When answering questions, always include relevant information from the question in your response."
    
    ##### <prompt the user with the estimated total cost before the first iteration> #####
    if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-instruct'] and using_prev_file == 0:
        assert opt.api_key != None, "Please specify your OpenAI API key."
        openai.api_key = opt.api_key
        estimated_cost, _ = estimate_cost(max_index - min_index, system_instruction, model, data, max_tokens, include_contexts=False, using_batch_api=using_batch_api, question_key=question_key)
        res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
        assert res.strip().lower() == 'y', "User decided to abort the process."
    ##### </prompt the user with the estimated total cost before the first iteration> #####
    
    ##### <using batch api for OpenAI models> #####
    if using_batch_api == 1 and using_prev_file == 0:
        samples = []
        for sample_index in trange(min_index, max_index):
            id = data[sample_index][id_key] if id_key != None else sample_index
            question = data[sample_index][question_key]
            body = {
                "model": model,
                "messages": [{
                    "role": "system",
                    "content": system_instruction
                },{
                    "role": "user",
                    "content": question
                }],
                "max_tokens": max_tokens,
                "logprobs": True,
                "top_logprobs": 20,
                "temperature": temperature,
                "top_p": model_top_p,
            }
            samples.append({
                "custom_id": str(id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            })
        batch_file_dir = f"tmp/{dataset}/{model}/batch_files/"
        if not os.path.exists(batch_file_dir):
            os.makedirs(batch_file_dir)
        batch_file_path = f"tmp/{dataset}/{model}/batch_files/{dataset}-{model}-{filetype}-{str(segment)+'-' if segment != None else ''}LAG.jsonl"
        with open(batch_file_path, 'w') as file:
            for sample in samples:
                file.write(json.dumps(sample) + '\n')
        os.environ["OPENAI_API_KEY"] = openai.api_key
        client = OpenAI()
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
            "description": f"{dataset}-{model}-{filetype}-{str(segment)+'-' if segment != None else ''}LAG"
            }
        )
        print(f"batch_id: {batch_object.id}") # BATCH FILE -|--------------------------------------------------------->>>>
    elif using_batch_api == 1 and using_prev_file == 1:
        samples = {}
        client = OpenAI(api_key=opt.api_key)
        output_file_id = client.batches.retrieve(batch_id).output_file_id
        file_response = client.files.content(output_file_id)
        lines = file_response.text.splitlines()
        list_of_dicts = [json.loads(line) for line in lines]
        nlp = spacy.load("en_core_web_md")
        for raw_response, sample_index in zip(list_of_dicts, trange(min_index, max_index)):
            response = raw_response['response']
            id = data[sample_index][id_key] if id_key != None else sample_index
            question = data[sample_index][question_key]

            question_entities = get_entities(question, nlp)

            datum = data[sample_index]
            answers = []
            for answer_key in answer_keys:
                if type(datum[answer_key]) == list:
                    answers.extend(datum[answer_key][0]["spans"])
                elif type(datum[answer_key]) == str:
                    answers.append(datum[answer_key])
                elif type(datum[answer_key]) == bool:
                    answers.append(datum[answer_key])
                else:
                    assert 1 == 0, 'An answer found that is neither a string nor a list nor boolean.'
            if v:
                print(f"Q: {highlight_ngrams(question, question_entities, highlight_all=False)}")
                print(f"A: {answers}")
                print(highlight_ngrams(f"{model} is responding...", color='green'))
            generated_answer = response['body']['choices'][0]['message']['content']
            generated_answer_entities = get_entities(generated_answer, nlp)

            if v:
                print('generated answer:')
                print(generated_answer)
                print('generated answer entities:')
                print(' '.join(generated_answer_entities))
                print(highlight_ngrams(generated_answer, generated_answer_entities, highlight_all=False))
            
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

            filtered_words = filter_words(words, word_entropies, dependent_words, question_entities, percentile)

            if v:
                for key, value in filtered_words.items():
                    print(f"template query (th={key}): {' '.join(value)}\n")
            
            for key, value in filtered_words.items():
                template_query = question + ' ' + ' '.join(value).strip() # 100 = GAR
                new_sample = {'id': id, 'question_org': question, 'question': template_query, 'answer': answers, 'generated_answer': generated_answer, 'word_level_entropies': list(zip(words, list(map(lambda x: round(x, 3), word_entropies)))), 'question_entities': question_entities, 'answer_entities': generated_answer_entities, 'dependent_words': dependent_words, 'word_entropies_median': round(np.percentile(word_entropies, 50), 3)}
                samples.setdefault(key, []).append(deepcopy(new_sample))
            new_sample = {'id': id, 'question_org': question, 'question': generated_answer, 'answer': answers, 'generated_answer': generated_answer, 'word_level_entropies': list(zip(words, list(map(lambda x: round(x, 3), word_entropies)))), 'question_entities': question_entities, 'answer_entities': generated_answer_entities, 'dependent_words': dependent_words, 'word_entropies_median': round(np.percentile(word_entropies, 50), 3)}
            samples.setdefault('x', []).append(deepcopy(new_sample)) # x: query = long-form answer

        for percent in percentile + ['x']:
            # if temperature == 0 and model_top_p == .1:
            #     file_path = f"new_results/{dataset}/{model}/template_queries_{dataset}_{filetype}_{model}_{percent}.jsonl"
            # else:
            if k_fold:
                file_path = f"runs/hints/{dataset}/{model}/{filetype}_long_P{percent}_T{temperature}_{k_fold}.jsonl"
            else:
                file_path = f"runs/hints/{dataset}/{model}/{filetype}_long_P{percent}_T{temperature}.jsonl"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, 'w' if segment == None or segment == 0 else 'a') as file:
                for sample in samples[percent]:
                    file.write(json.dumps(sample) + '\n')
    ##### </using batch api for OpenAI models> #####

    if using_batch_api == 0:
        samples = {}
        nlp = spacy.load("en_core_web_md")
        for sample_index in trange(n):
            id = data[sample_index][id_key] if id_key != None else sample_index
            question = data[sample_index][question_key]
            question_entities = get_entities(question, nlp)
            datum = data[sample_index]
            answers = []
            for answer_key in answer_keys:
                if type(datum[answer_key]) == list:
                    answers.extend(datum[answer_key])
                elif type(datum[answer_key]) == str:
                    answers.append(datum[answer_key])
                else:
                    assert 1 == 0, 'An answer found that is neither a string nor a list.'
            if v:
                print(f"Q: {highlight_ngrams(question, question_entities, highlight_all=False)}")
                print(f"A: {answers}")
                print(highlight_ngrams(f"{model} is responding...", color='green'))
                tic = time()
            if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "system",
                        "content": system_instruction
                    },{
                        "role": "user",
                        "content": question
                    }],
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=20,
                    temperature=temperature,
                    top_p=model_top_p,
                )
                generated_answer = response.choices[0].message.content
            elif model in ['gpt-3.5-turbo-instruct']:
                client = OpenAI(api_key=opt.api_key)
                response = client.completions.create(
                    model=model,
                    prompt=system_instruction + '\nQuestion: ' + question,
                    logprobs=20,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                generated_answer = response.choices[0].text.strip()
                # print(response.choices[0].logprobs)
            elif model in ['llama-3']:
                generated_answer, meta_top_probs = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, system_instruction, template_query_generation=True)
            if v:
                toc = time()
                print(highlight_ngrams(f"response time: {(toc - tic):.2f} s", color='blue'))
            generated_answer_entities = get_entities(generated_answer, nlp)
            if v:
                print('generated answer:')
                print(generated_answer)
                print('generated answer entities:')
                print(' '.join(generated_answer_entities))
                print(highlight_ngrams(generated_answer, generated_answer_entities, highlight_all=False))

            token_probabilities, possible_tokens, generated_tokens, generated_tokens_indices, top_p = [], [], [], [], []

            x = 0 # num_answers is set to 1 by default
            if model in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
                for j in range(len(response.choices[x].logprobs.content)):
                    top_20_logprobs = response.choices[x].logprobs.content[j].top_logprobs
                    token_probabilities.append(list(map(lambda x: np.exp(x.logprob), top_20_logprobs)))
                    possible_tokens.append(list(map(lambda x: x.token, top_20_logprobs)))
                    generated_token = response.choices[x].logprobs.content[j].token
                    generated_tokens.append(generated_token)
                    try:
                        generated_token_index = list(map(lambda x: x.token, top_20_logprobs)).index(generated_token)
                        generated_tokens_indices.append(generated_token_index)
                        top_p.append(token_probabilities[0][generated_token_index])
                    except:
                        generated_tokens_indices.append(None)
                        top_p.append(None)
            elif model in ['gpt-3.5-turbo-instruct']:
                choice = response.choices[x]
                logprobs = choice.logprobs
                for j, top_logprob_dict in enumerate(logprobs.top_logprobs):
                    token_probs = [np.exp(lp) for lp in top_logprob_dict.values()]
                    tokens = list(top_logprob_dict.keys())
                    generated_token = logprobs.tokens[j]

                    token_probabilities.append(token_probs)
                    possible_tokens.append(tokens)
                    generated_tokens.append(generated_token)
            elif model in ['llama-3']:
                for j in range(len(meta_top_probs)):
                    top_20_probs = meta_top_probs[j]['top_probs']
                    token_probabilities.append(list(map(lambda x: x['prob'], top_20_probs)))
                    possible_tokens.append(list(map(lambda x: x['token'], top_20_probs)))
                    generated_token = meta_top_probs[j]['token']
                    generated_tokens.append(generated_token)
                    try:
                        generated_token_index = list(map(lambda x: x['token'], top_20_probs)).index(generated_token)
                        generated_tokens_indices.append(generated_token_index)
                        top_p.append(token_probabilities[0][generated_token_index])
                    except:
                        generated_tokens_indices.append(None)
                        top_p.append(None)

            entropies = get_token_level_entropies(token_probabilities)
            words, word_entropies = get_word_level_entropies(generated_tokens, entropies)
            word_entropies, dependent_words = combine_dependent_words_entropies(words, deepcopy(word_entropies), generated_answer_entities)
            if v:
                # for i, prob in enumerate(top_p):
                #     print(f"P for token {i + 1} ({generated_tokens[i].strip()}): {prob:.4f}")
                # print('-'*30)
                # for i, entropy in enumerate(entropies):
                #     print(f"Entropy for token {i + 1} ({generated_tokens[i].strip()}): {entropy:.4f}")
                # print('-'*30)
                # for i, entropy in enumerate(word_entropies):
                #     print(f"Entropy for word {i + 1} ({words[i].strip()}): {entropy:.4f}")
                # for i, p in enumerate(top_p):
                #     print(f"P for token {i + 1} ({generated_tokens[i].strip()}): {p:.4f}")
                # colorized_sentence = colorize_tokens(words, word_entropies)
                # display(HTML(colorized_sentence))
                pass

            filtered_words = filter_words(words, word_entropies, dependent_words, question_entities, percentile)
            if v:
                for key, value in filtered_words.items():
                    print(f"template query (th={key}): {' '.join(value)}\n")
            for key, value in filtered_words.items():
                template_query = question + ' ' + ' '.join(value).strip() # 100 = GAR
                new_sample = {'id': id, 'question_org': question, 'question': template_query, 'answer': answers, 'generated_answer': generated_answer, 'word_level_entropies': list(zip(words, list(map(lambda x: round(x, 3), word_entropies)))), 'question_entities': question_entities, 'answer_entities': generated_answer_entities, 'dependent_words': dependent_words, 'word_entropies_median': round(np.percentile(word_entropies, 50), 3)}
                samples.setdefault(key, []).append(deepcopy(new_sample))
            new_sample = {'id': id, 'question_org': question, 'question': generated_answer, 'answer': answers, 'generated_answer': generated_answer, 'word_level_entropies': list(zip(words, list(map(lambda x: round(x, 3), word_entropies)))), 'question_entities': question_entities, 'answer_entities': generated_answer_entities, 'dependent_words': dependent_words, 'word_entropies_median': round(np.percentile(word_entropies, 50), 3)}
            samples.setdefault('x', []).append(deepcopy(new_sample)) # x: query = long-form answer

        for percent in percentile + ['x']:
            # if temperature == 0 and model_top_p == .1:
            #     file_path = f"new_results/{dataset}/{model}/template_queries_{dataset}_{filetype}_{model}_{percent}.jsonl"
            # else:
            if k_fold:
                file_path = f"runs/hints/{dataset}/{model}/{filetype}_long_P{percent}_T{temperature}_{k_fold}.jsonl"
            else:
                file_path = f"runs/hints/{dataset}/{model}/{filetype}_long_P{percent}_T{temperature}.jsonl"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, 'w' if segment == None or segment == 0 else 'a') as file:
                for sample in samples[percent]:
                    file.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help="Path to the data")
    parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Model name: gpt-3.5-turbo, gpt-4o, llama-3 (llama3-ChatQA-1.5-8B), gpt-4o-mini, gpt-3.5-turbo-instruct")
    parser.add_argument('--api_key', required=False, type=str, default=None, 
                        help="OpenAI's API key")
    parser.add_argument('--max_tokens', required=False, type=int, default=50, 
                        help="Maximum number of the output tokens")
    parser.add_argument('--temperature', required=False, type=float, default=0, 
                        help="Model config")
    parser.add_argument('--k_fold', required=False, type=int, default=None, 
                        help="1, 2, 3, or 4")
    parser.add_argument('--top_p', required=False, type=float, default=1, 
                        help="Model config")
    parser.add_argument('--using_batch_api', required=False, type=int, default=1, 
                        help="Set to 0 if not using the Batch API")
    parser.add_argument('--using_prev_file', required=False, type=int, default=0, 
                        help="Set to 1 if using a previously created file")
    parser.add_argument('--batch_id', required=False, type=str, default=None, 
                        help="Batch ID")
    parser.add_argument('--filetype', required=False, type=str, default='test', 
                        help="train | test | dev")
    parser.add_argument('--dataset', required=False, type=str, default=None, 
                        help="MuSiQue, IIRC, HotpotQA, 2WikiMultihopQA, and StrategyQA")
    parser.add_argument('--percentile', required=False, type=int, nargs='+', default=[], 
                    help="0: ignore_entropies = True | 100: template_query = question + answer | 50: named_entities_masking_threshold = median(all_entropies) | x: only the long-form answer")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")
    parser.add_argument('--segment', required=False, type=int, default=None, 
                        help=f"If the input file size is > {BATCH_FILE_LENGTH_LIMIT} you have to run it multiple times, each time on a specific segment.")

    args = parser.parse_args()
    main(args)
