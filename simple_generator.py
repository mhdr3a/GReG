import json
import openai
import sys
import re
from unidecode import unidecode

def normalize_and_compare(str1, str2):
    str1_normalized = re.sub(r'[^\w\d]', '', unidecode(re.sub(r'\b(a|an|the)\b', '', str1, flags=re.IGNORECASE)).lower())
    str2_normalized = re.sub(r'[^\w\d]', '', unidecode(re.sub(r'\b(a|an|the)\b', '', str2, flags=re.IGNORECASE)).lower())
    return str1_normalized == str2_normalized

# Load your dataset
dataset_path = sys.argv[1]
with open(dataset_path, 'r') as file:
    data = [json.loads(line) for line in file]

retrieved_data_path = sys.argv[2]
with open(retrieved_data_path, 'r') as file:
    retrieved_data = json.load(file)

openai.api_key = sys.argv[3]

base_model = sys.argv[4]

def generate_answer(question):
    prompt = f"Your task is to answer this question into 2 to 3 words and in a format of factoid answer. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers. \nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

results = []
results = []
for i, item in enumerate(data):
    question = item['question_org']
    gold_answer = item['answer_org']
    
    context_texts = [doc['text'] for doc in retrieved_data[i]['ctxs'][:4]]

    generated_answer = generate_answer(question, context_texts)
    
    exact_match = normalize_and_compare(gold_answer, generated_answer)

    results.append({
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "exact_match": exact_match
    })
    
    print(f"Processed item {i+1}/{len(data)}")
    
exact_match_ratio = sum(result['exact_match'] for result in results) / len(results)

for result in results:
    print(f"Question: {result['question']}")
    print(f"Gold Answer: {result['gold_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print(f"Exact Match: {result['exact_match']}")
    print()

print(f"Exact Match Ratio: {exact_match_ratio:.2f}")

results_file_path = sys.argv[5]
with open(results_file_path, 'w') as results_file:
    json.dump(results, results_file, indent=4)
