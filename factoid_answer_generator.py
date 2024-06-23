import argparse
import re
from unidecode import unidecode
import json
from tqdm import trange
import openai
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertModel
import numpy as np
from termcolor import colored
import os

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
    str1_normalized = re.sub(r'[^\w\d]', '', unidecode(re.sub(r'\b(a|an|the)\b', '', str1, flags=re.IGNORECASE)).lower())
    return str1_normalized

def compare(str1, str2):
    return str1 == str2

def estimate_cost(total_samples, fixed_instructions_msg, model, questions, contexts, max_output_tokens_per_sample, word_to_token_ratio=1.33):
    fixed_instructions_tokens = word_to_token_ratio * len(fixed_instructions_msg.replace('\n', ' ').strip().split())
    total_context_tokens = word_to_token_ratio * sum(list(map(lambda x: sum(list(map(lambda y: len(y.replace('\n', ' ').strip().split()), x))), contexts)))
    total_input_tokens = word_to_token_ratio * sum(list(map(lambda q: len(q['question'].replace('\n', ' ').strip().split()), questions))) + total_samples * fixed_instructions_tokens + total_context_tokens
    total_output_tokens = total_samples * max_output_tokens_per_sample
    cost_per_1M_tokens_input = 5 if model == 'gpt-4o' else (.5 if model == 'gpt-3.5-turbo' else (10 if model == 'gpt-4-turbo' else None))
    cost_per_1M_tokens_output = 15 if model == 'gpt-4o' else (1.5 if model == 'gpt-3.5-turbo' else (30 if model == 'gpt-4-turbo' else None))

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

def evaluate_saved_file(opt):
    filepath = opt.data
    data = read_jsonl(filepath)
    n = len(data)
    metrics = opt.metrics # 0: Exact Match and ROUGE | 1: Lenient and Strict Accuracy
    file_metrics = int(filepath.split('/')[-1].split('_')[4])
    assert file_metrics == metrics, "metrics argument does not match the saved file's metrics."
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Only works if metrics = 0
    EM = 0; precision = 0; recall = 0; fmeasure = 0; sem_sim = 0 # Only works if metrics = 0
    Sacc = 0; Lacc = 0 # Only works if metrics = 1
    print(f'\n# of samples: {n}')
    for i in trange(n):
        datum = data[i]
        answer = datum['gold_answer']
        generated_answer = datum['pred_answer']
        mets = evaluate_factoid_answers(metrics, answer, generated_answer, scorer)
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

def add_contexts(full_prompt, contexts, top_k, include_titles):
    for j in range(top_k):
        title = contexts[j]['title']; text = contexts[j]['text']
        tmp = "\nText: "
        full_prompt += f"{j + 1}. {f'Title: {title}{tmp}' if include_titles else ''}{text}"
        full_prompt += '\n'
        if j != top_k - 1:
            full_prompt += '\n'
    return full_prompt

def openai_generator(api_key, model, full_prompt, max_tokens, num_answers, temperature, model_top_p):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": full_prompt
        }],
        max_tokens=max_tokens,
        n=num_answers,
        temperature=temperature,
        top_p=model_top_p,
    )
    return response.choices[0].message.content

def evaluate_factoid_answers(metrics, answer, generated_answer, scorer):
    EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc = 0, 0, 0, 0, 0, 0, 0
    if metrics == 0 or metrics == 2: # Exact Match (EM) and ROUGE
        EM = compare(normalize(answer), normalize(generated_answer))
        scores = scorer.score(answer, generated_answer)['rougeL']
        precision = scores.precision; recall = scores.recall; fmeasure = scores.fmeasure
    elif metrics == 1: # Sacc and Lacc (Strict and Lenient Accuracy)
        factoids = json.loads(correct_json_list(generated_answer))
        Sacc = compute_strict_accuracy(normalize(answer), list(map(normalize, factoids)))
        Lacc = compute_lenient_accuracy(normalize(answer), list(map(normalize, factoids)))
    if metrics == 2: # Semantic Similarity
        sem_sim = compute_semantic_similarity(answer, generated_answer)
    return EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    dataset = filepath.split('/')[-3]
    assert dataset in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA', 'NQ', 'TriviaQA']
    
    data, id_key, n = read_data(filepath, ext, dataset)
    
    model = opt.model
    assert model in ['gpt-3.5-turbo', 'gpt-4o']
    v = int(opt.verbose)
    assert v == 0 or v == 1
    max_tokens = int(opt.max_tokens)
    assert max_tokens > 0
    num_answers = opt.num_answers
    assert num_answers > 0
    temperature = opt.temperature
    assert 0 <= temperature <= 1
    model_top_p = opt.top_p
    assert .1 <= model_top_p <= 1
    api_key = opt.api_key
    top_k = opt.top_k
    assert 0 <= top_k <= 10
    include_titles = opt.include_titles
    assert include_titles == 0 or include_titles == 1
    metrics = opt.metrics # 0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity
    assert metrics == 0 or metrics == 1 or metrics == 2
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Only works if metrics = 0
    EM = 0; precision = 0; recall = 0; fmeasure = 0; # Only works if metrics = 0 | 2
    sem_sim = 0 # Only works if metrics = 2
    Sacc = 0; Lacc = 0 # Only works if metrics = 1
    samples = []
    first_iter = True
    
    for i in trange(3):
        datum = data[i]
        question = datum['question']
        answer = datum['answer']
        id = datum[id_key]
        ##### <defining the prompt> #####
        full_prompt = ''; initial_prompt = ''
        if top_k > 0: # include_contexts = True
            if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
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
            estimated_cost, _ = estimate_cost(n, initial_prompt if top_k > 0 else full_prompt, model, data, list(map(lambda x: list(map(lambda y: y['title'] + '\n' + y['text'], x['ctxs'][:top_k])), data)), max_tokens)
            res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
            assert res.strip().lower() == 'y', "User decided to abort the process."
            first_iter = False
        ##### </prompt the user with the estimated total cost before the first iteration> #####
        generated_answer = openai_generator(api_key, model, full_prompt, max_tokens, num_answers, temperature, model_top_p)

        mets = evaluate_factoid_answers(metrics, answer, generated_answer, scorer)
        EM += mets[0]; precision += mets[1]; recall += mets[2]; fmeasure += mets[3]; sem_sim += mets[4]; Sacc += mets[5]; Lacc += mets[6]

        samples.append({'id': id, 'question': question, 'gold_answer': answer, 'pred_answer': generated_answer})
        ##### <printing logs> #####
        if v:
            print(f'\nP: {highlight_ngrams(full_prompt, color="blue")}')
            color = ''
            if metrics == 0:
                color = 'green' if mets[0] else 'red'
            else:
                color = 'green' if mets[6] else 'red'
            print(f'A: {answer}\nO: {highlight_ngrams(generated_answer, color=color)}')
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
    file_path = f'results/{dataset}/{model}/{dataset}_{model}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
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
    parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Exact name of the model")
    parser.add_argument('--top_k', required=False, type=int, default=0, 
                        help="Top-k contexts used for Retrieval-Augmented Generation (RAG)")
    parser.add_argument('--include_titles', required=False, type=int, default=1, 
                        help="0: Do not include the titles for the contexts | 1: Include the titles for the contexts | Only works when top_k > 0")
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
    parser.add_argument('--metrics', required=False, type=int, default=0, 
                        help="0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity")
    parser.add_argument('--eval_only', required=False, type=int, default=0, 
                        help="0: Generate factoid answers, then evaluate | 1: Only evaluate a saved file")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")

    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_saved_file(args)
    else:
        main(args)
