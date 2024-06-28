import json
import argparse

class InvalidFileExtensionError(Exception):
    def __init__(self, extension, message="Invalid file extension"):
        self.extension = extension
        self.message = f"{message}: {extension}"
        super().__init__(self.message)

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

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    file_type = filepath.split('.')[-2]
    dataset = 'IIRC'
    data, _, _ = read_data(filepath, ext)
    no_ans = 0; ans = 0
    preprocessed = []
    for x in data:
        for y in x['questions']:
            try:
                answer_spans = y['answer']['answer_spans']
                answer = ' '.join(list(map(lambda h: h['text'], answer_spans))).strip()
                question = y['question']
                preprocessed.append({'id': ans, 'question': question, 'answer': answer})
                ans += 1
            except:
                no_ans += 1
    with open(f'{file_type}_preprocessed.json', 'w') as f:
        json.dump(preprocessed, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', required=True, type=str, default=None, 
                        help="Path to the data")
    args = parser.parse_args()
    
    main(args)
