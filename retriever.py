import dspy
import json
from copy import deepcopy
from tqdm import trange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--top_k', required=False, type=int, default=5, 
                    help="top-k retrieved passages per query")
args = parser.parse_args()
top_k = args.top_k

def retrieve(query: str, k: int):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=k)
    return [x['text'] for x in results]

dataset_names = ['HotpotQA', '2WikiMultihopQA', 'MuSiQue']

for dataset_name in dataset_names:
    queries_path = f'runs/hints/{dataset_name}/gpt-4o/best_queries.jsonl'
    samples = []
    print(f"Retrieving for {dataset_name}...")
    with open(queries_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    results = []
    for i in trange(len(samples)):
        sample = samples[i]
        tmp = deepcopy(sample)
        while True:
            try:
                passages = retrieve(sample['question_org'], top_k)
                tmp['context_org'] = deepcopy(passages)
                break
            except Exception as e:
                print(e)
                continue
        while True:
            try:
                passages = retrieve(sample['question'], top_k)
                tmp['context'] = deepcopy(passages)
                results.append(deepcopy(tmp))
                break
            except:
                continue

    with open(queries_path, 'w', encoding='utf-8') as f:
        for d in results:
            f.write(json.dumps(d) + '\n')
    print(f"Retrieval is done for {dataset_name}.")
