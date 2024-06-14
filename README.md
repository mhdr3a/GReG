# GReG
Generative Retrieval Generator for Multi-hop Question Answering

MuSiQue Dataset:
```
git clone https://github.com/stonybrooknlp/musique
bash /content/musique/download_data.sh
pip install openai
```

HotpotQA Dataset:
```
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

Wrap the output content of each notebook cell:
```
from IPython.display import display, HTML
def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)
```

Read a jsonl file:
```
import json

def read_jsonl(filepath='/content/data/musique_ans_v1.0_dev.jsonl'):
  data = []
  with open(filepath, 'r') as file:
    for line in file:
      data.append(json.loads(line.strip()))
  return data
data = read_jsonl()
```

IIRC Dataset:
```
data = []
for x in read_json():
  for y in x['questions']:
    try:
      answer_spans = y['answer']['answer_spans']
      answer = ' '.join(list(map(lambda h: h['text'], answer_spans)))
      question = y['question']
      data.append({'question': question, 'answer': answer})
    except:
      pass
```
