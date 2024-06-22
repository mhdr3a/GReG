import argparse
import openai
import numpy as np
from IPython.display import display, HTML
from time import time
import re
import spacy
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy
from termcolor import colored
import math
import json

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

def highlight_ngrams(sentence, ngrams_to_highlight=[], color='red', highlight_all=True):
    if highlight_all:
        return colored(sentence, color)
    ngrams_to_highlight = sorted(ngrams_to_highlight, key=lambda x: len(x.split()), reverse=True)
    for ngram in ngrams_to_highlight:
        sentence = sentence.replace(ngram, colored(ngram, color))
    return sentence

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
    pattern = r'[.,;:!?]'
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
    for n in range(1, max_window_size + 1):
        found_ngrams = locate_ngrams(words, n, output_entities)
        for ngram in found_ngrams:
            ents = []
            for i in ngram[1]:
                ents.append(word_entropies[i])
                max_ents = max(ents)
            for i in ngram[1]:
                word_entropies[i] = max_ents
    return word_entropies

def filter_words(words, word_entropies, question_entities, output_entities, threshold, mask=' '):
    th = np.percentile(word_entropies, threshold) # threshold = 0 -> ignore_entropies = True, threshold = 100 -> template_query = question
    filtered_words = []
    for word, entropy in zip(words, word_entropies):
        norm_word = remove_punctuation(word)
        if (entropy > th) and any(norm_word in ent for ent in output_entities) and (not any(norm_word in ent for ent in question_entities)):
            filtered_words.append(mask)
            continue
        filtered_words.append(word)
    return filtered_words

def get_entities(sentence):
    entities = []
    nlp = spacy.load("en_core_web_md")
    doc = nlp(sentence)
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

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

def iirc_preprocess(d):
    no_ans = 0; ans = 0
    data = []
    for x in d:
        for y in x['questions']:
            try:
                answer_spans = y['answer']['answer_spans']
                answer = ' '.join(list(map(lambda h: h['text'], answer_spans)))
                question = y['question']
                data.append({'question': question, 'answer': answer})
                ans += 1
            except:
                no_ans += 1
    return data

class InvalidFileExtensionError(Exception):
    def __init__(self, extension, message="Invalid file extension"):
        self.extension = extension
        self.message = f"{message}: {extension}"
        super().__init__(self.message)

def read_data(filepath, ext, dataset):
    data = []
    if ext == 'json':
        data = read_json(filepath)
    elif ext == 'jsonl':
        data = read_jsonl(filepath)
    else:
        raise InvalidFileExtensionError(ext)
    id_key = None
    if dataset == 'IIRC':
        pass
    elif dataset == 'MuSiQue':
        id_key = 'id'
    else: # HotpotQA and 2WikiMultihopQA
        id_key = '_id'
    n = len(data)
    return data, id_key, n

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    dataset = filepath.split('/')[-2] + '_' + filepath.split('/')[-1].split('.')[0]
    assert dataset in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA', 'NQ', 'TriviaQA']
    
    data, id_key, n = read_data(filepath, ext, dataset)
    
    model = opt.model
    v = opt.verbose
    max_tokens = opt.max_tokens
    num_answers = opt.num_answers
    percentile = opt.percentile
    assert int(percentile) in [0, 50, 100]
    temperature = opt.temperature
    model_top_p = opt.top_p
    
    samples = []
    openai.api_key = opt.api_key
    for sample_index in trange(n):
        id = data[sample_index][id_key] if id_key != None else None
        question = data[sample_index]['question']
        question_entities = get_entities(question)
        answer = data[sample_index]['answer']
        if v:
            print(f"Q: {highlight_ngrams(question, question_entities, highlight_all=False)}")
            print(f"A: {answer}")
            print(highlight_ngrams(f"{model} is responding...", color='green'))
            tic = time()
        response = openai.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "When answering questions, always include relevant information from the question in your response."
            },{
                "role": "user",
                "content": question
            }],
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=20,
            n=num_answers,
            temperature=temperature,
            top_p=model_top_p,
        )
        if v:
            toc = time()
            print(highlight_ngrams(f"response time: {(toc - tic):.2f} s", color='blue'))
        generated_answer = response.choices[0].message.content
        generated_answer_entities = get_entities(generated_answer)
        if v:
            print(highlight_ngrams(generated_answer, generated_answer_entities, highlight_all=False))

        token_probabilities, possible_tokens, generated_tokens, generated_tokens_indices, top_p = [], [], [], [], []

        for x in range(num_answers):
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

            entropies = get_token_level_entropies(token_probabilities)
            words, word_entropies = get_word_level_entropies(generated_tokens, entropies)
            word_entropies = combine_dependent_words_entropies(words, deepcopy(word_entropies), generated_answer_entities)
            if v:
                # for i, entropy in enumerate(entropies):
                #     print(f"Entropy for token {i + 1} ({generated_tokens[i].strip()}): {entropy:.4f}")
                # for i, p in enumerate(top_p):
                #     print(f"P for token {i + 1} ({generated_tokens[i].strip()}): {p:.4f}")
                # colorized_sentence = colorize_tokens(words, word_entropies)
                # display(HTML(colorized_sentence))
                pass

            filtered_words = filter_words(words, word_entropies, question_entities, generated_answer_entities, percentile)
            if v:
                print(f"template query: {' '.join(filtered_words)}\n")
            new_sample = {'id': id, 'question_org': question, 'question': ' '.join(filtered_words).strip(), 'answer_org': answer, 'answer': 'None'}
            samples.append(new_sample)
    with open(f"template_queries/{dataset}_{model}_{percentile}.jsonl", 'w') as file:
        for item in samples:
            file.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help="Path to the data")
    parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Exact name of the model")
    parser.add_argument('--api_key', required=False, type=str, default='sk-proj-GoCByQuK462EuWXCXjEkT3BlbkFJD8h4uoNag5STZiKJdpN7', 
                        help="OpenAI API key")
    parser.add_argument('--max_tokens', required=False, type=int, default=50, 
                        help="Maximum number of the output tokens")
    parser.add_argument('--num_answers', required=False, type=int, default=1, 
                        help="Number of answers generated by the model")
    parser.add_argument('--temperature', required=False, type=float, default=0, 
                        help="Model config")
    parser.add_argument('--top_p', required=False, type=float, default=.1, 
                        help="Model config")
    parser.add_argument('--percentile', required=False, type=int, default=50, 
                        help="0: ignore_entropies = True | 100: template_query = question | 50: named_entities_masking_threshold = median(all_entropies)")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")

    args = parser.parse_args()
    main(args)
