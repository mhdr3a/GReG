# GReG

<table style="border: none;">
  <tr>
    <td valign="top" width="80%">
      <h2>A Generate-Retrieve-Generate Framework for Open Domain Question Answering</h2>
      <p>The <b>G</b>enerate-<b>Re</b>trieve-<b>G</b>enerate (GReG) Framework enhances state-of-the-art generative models by incorporating helpful retrieved passages, thereby improving the accuracy and confidence of answers to open-domain multi-hop questions while reducing hallucinations. Initially, the generative model provides an answer based on a simple prompt. A classifier then determines if additional context is needed, based on the entropy of the generated tokens. If more context is required, the framework constructs a query using the question and the initial answer, retrieves relevant passages from Wikipedia, and re-prompts the generative model with a factoid answer-requesting prompt. The model's performance in open-domain QA is evaluated using Exact Match (EM) and ROUGE metrics.</p>
    </td>
    <td valign="top" width="20%">
      <img src="img/greg.png" width="100%">
    </td>
  </tr>
</table>

Initial Setup:
```
conda create --name greg python=3.8
conda activate greg
conda install pip
git clone https://github.com/facebookresearch/FiD.git
pip install "pydantic>=1.7.4,<3.0.0"
cd FiD
pip install -r requirements.txt
pip install filelock
pip install typing-extensions
bash get-data.sh
bash get-model.sh -m nq_retriever
conda install six
python  generate_passage_embeddings.py \
        --model_path pretrained_models/nq_retriever \
        --passages open_domain_data/psgs_w100.tsv \
        --output_path wikipedia_embeddings \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
pip install gdown
pip install openai
pip install IPython
pip install spacy
python -m spacy download en_core_web_md
pip install matplotlib
pip install termcolor
pip install unidecode
pip install rouge-score
cd ..
git clone https://github.com/mhdr3a/GReG.git
cd GReG
bash download_datasets.sh # MuSiQue, HotpotQA, IIRC, and 2WikiMultihopQA
```

Replace ```FiD/src/data.py``` with ```GReG/src/data.py```

Replace ```FiD/src/evaluation.py``` with ```GReG/src/evaluation.py```

```
<dataset_name> = [MuSiQue, HotpotQA, IIRC, 2WikiMultihopQA, NQ, TriviaQA]
<model_name> = [gpt-3.5-turbo, gpt-4o]
<temperature> = [0, 1]
<top_p> = [0.1, 1]
```

Passage retrieval using a pre-trained DPR on NQ (percentile = 100: query = question, 50: template query considering entropies, 0: template query ignoring entropies):
```
python ../FiD/passage_retrieval.py \
    --model_path ../FiD/pretrained_models/nq_retriever \
    --passages ../FiD/open_domain_data/psgs_w100.tsv \
    --data results/<dataset_name>/<model_name>/<dataset_name>_<model_name>_<0,50>.jsonl \
    --passages_embeddings ../FiD/wikipedia_embeddings_00 \
    --output_path results/<dataset_name>/percentile_<0,50,100>/retrieved_passages_<model_name>.json \
    --n-docs 10 \
```

Template query generation in 2 ways (percentile = 50: considering the entropies median as a threshold to mask the named entities | 0: masking all the named entities present in the answer yet absent in the question):
```
python template_query_generator.py \
    --data data/<dataset_name>/<filename> \
    --model <model_name> \
    --api_key <your_openai_api_key> \
    --percentile <0,50>
```

Factoid answer generation using 4 (top_k > 0: with context | top_k = 0: without context) different prompts (metrics are either 0:(EM, ROUGE-F1, Semantic Similarity) or 1:(Sacc, Lacc)):
```
python factoid_answer_generator.py \
      --data results/<dataset_name>/percentile_<0,50,100>/retrieved_passages_<model_name>.json \
      --model <model_name> \
      --top_k <0,1,5,10> \
      --api_key <your_openai_api_key> \
      --metrics <0,1>
```

## Llama Setting
To run Factoid answer generation for llama 3, set the huggingface access token with the following command:
```
export HF_TOKEN="your_hugging_face_token"
```
and make sure to set the temperature is greater than 0. 
