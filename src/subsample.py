import json

data_paths = {'HotpotQA':        'data/HotpotQA/hotpot_dev_distractor_v1.json',
              '2WikiMultihopQA': 'data/2WikiMultihopQA/dev.json',
              'MuSiQue':         'data/MuSiQue/musique_ans_v1.0_dev.jsonl'}

for key, value in data_paths.items():
    print(f"Subsampling for {key}...")
    data = []
    data_path = value
    if data_path[-1] == 'n':
        # Load the original JSON file
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        # Load the original JSONL file
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    
    # Take the first 500 samples
    subset = data[:500]
    
    # Save to a JSONL file
    if data_path[-1] == 'l':
        data_path = data_path[:-1]
    with open(data_path.replace('.json', '_sampled.jsonl'), 'w') as f:
        for item in subset:
            f.write(json.dumps(item) + '\n')
    print(f"{key} is subsampled.")
