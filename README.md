# GReG

<table style="border: none;">
  <tr>
    <td valign="top" width="80%" style="border-width: 0 !important;">
      <h2>Pre-Generative Retrieval-Augmented Generator for Multi-hop Question Answering</h2>
      <p>The Pre-<b>G</b>enerative <b>Re</b>trieval-Augmented <b>G</b>enerator (GReG) for multi-hop question answering is a sophisticated AI architecture designed to tackle complex queries that require synthesizing information from multiple sources or steps. This technology stands at the cutting edge of natural language processing and machine learning, offering significant improvements over traditional single-passage retrieval systems.</p>
    </td>
    <td valign="top" width="20%" style="border-width: 0 !important;">
      <img src="img/greg.png" width="100%">
    </td>
  </tr>
</table>

Initial Setup:
```
conda create --name greg python=3.9
conda activate greg
conda install pip
git clone https://github.com/mhdr3a/GReG.git
cd GReG
pip install -r requirements.txt
```

Loading the MuSiQue Dataset:
```
git clone https://github.com/stonybrooknlp/musique
bash /content/musique/download_data.sh
pip install openai
```

Loading the HotpotQA Dataset:
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

Loading the IIRC Dataset:
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

Installing the SpaCy Medium Language Core:
```
python -m spacy download en_core_web_md
```
Clone the FiD repo for the passage retrieval phase:
```
git clone https://github.com/facebookresearch/FiD.git
```

**Indexing Wikipedia Knowledge Source (Wiki Source Indexer)**

```
python  FiD/generate_retriever_embedding.py \
        --model_path <model_dir> \ #directory
        --passages passages.tsv \ #.tsv file
        --output_path wikipedia_embeddings \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
```

**Passage Retrieval**
Once indexing is complete, you can efficiently retrieve passages for a given input query:

```
python FiD/passage_retrieval.py \
    --model_path <model_dir> \
    --passages psgs_w100.tsv \
    --data_path data.json \
    --passages_embeddings "wikipedia_embeddings/wiki_*" \
    --output_path retrieved_data.json \
    --n-docs 100 \
```

Factoid Answers Exact Match Evaluation:
```
pip install unidecode
python [dataset_path] [retrieval_file_path] [OpenAI_API_Key] [base_model_name] [results_file_path]
```
