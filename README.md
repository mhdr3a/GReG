# GReG

<table style="border: none;">
  <tr>
    <td valign="top" width="80%">
      <h2>From Hints to Answers: Improving Multi-Hop QA with LLM-Guided Retrieval</h2>
      <p>Multi-hop question answering (QA) requires multiple reasoning steps to arrive at accurate factual answers. While pre-trained large language models (PLLMs) are increasingly used for multi-hop QA, their pre-training knowledge can be incomplete and lead to incorrect answers. Retrieval-augmented language modeling (RALM) addresses this limitation by retrieving external contexts from structured or unstructured knowledge sources. Existing RALM methods typically use the original multi-hop question for retrieval, but such questions often lack important intermediate facts, which results in insufficient evidence for precise reasoning. Meanwhile, LLMs may possess partial knowledge of the reasoning chain and can provide valuable "hints" that help guide the retrieval process. On the other hand, large-scale LLMs, despite their extensive parametric knowledge, are expensive and token-inefficient when handling lengthy retrieval-augmented input prompts. To address these challenges, we propose a Generate–Retrieve–Generate (GReG) pipeline, in which a large-scale LLM first generates an initial long-form answer that enriches the retrieval query and guides the retrieval of more relevant documents. A smaller and cost-effective model then leverages the retrieved context to produce the final factoid answer. Our experiments on widely used multi-hop QA datasets, HotpotQA and 2WikiMultihopQA, consistently demonstrate that GReG outperforms state-of-the-art multi-hop QA methods. We also show that GReG achieves a much better balance between accuracy and cost than simply relying on large-scale LLMs for the entire pipeline.</p>
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
wget https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_retriever.tar.gz
tar -xzvf tqa_retriever.tar.gz -C pretrained_models/
rm tqa_retriever.tar.gz
conda install six
python  generate_passage_embeddings.py \
        --model_path pretrained_models/nq_retriever \
        --passages open_domain_data/psgs_w100.tsv \
        --output_path wikipedia_embeddings_nq \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
python  generate_passage_embeddings.py \
        --model_path pretrained_models/tqa_retriever \
        --passages open_domain_data/psgs_w100.tsv \
        --output_path wikipedia_embeddings_tqa \
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
pip install huggingface-hub==0.23.4 pyyaml==6.0.1 safetensors==0.4.3 tokenizers==0.19.1 transformers==4.42.3
pip install accelerate==0.31.0 psutil==6.0.0
```

1. Replace ```FiD/src/data.py``` with ```GReG/src/data.py```

2. Replace ```FiD/src/evaluation.py``` with ```GReG/src/evaluation.py```


```
<dataset_name> = [MuSiQue, HotpotQA, IIRC, 2WikiMultihopQA]
<model_name> = [gpt-4o]
<retriever_name> = [nq]
<temperature> = [0, 1]
<top_p> = [0.1, 1]
```

Passage retrieval using a pre-trained DPR on NQ:
```
python ../FiD/passage_retrieval.py \
    --model_path ../FiD/pretrained_models/<retriever_name>_retriever \
    --passages ../FiD/open_domain_data/psgs_w100.tsv \
    --data results/<dataset_name>/<model_name>/template_queries_<dataset_name>_<model_name>.jsonl \
    --passages_embeddings ../FiD/wikipedia_embeddings_<retriever_name>_00 \
    --output_path results/<dataset_name>/retrieved_passages_<model_name>_<retriever_name>.json \
    --n-docs 10 \
```

Template query generation:
```
python template_query_generator.py \
    --data data/<dataset_name>/<model_name>/template_queries_<dataset_name>_<model_name>_<temperature>_<top_p>.jsonl \
    --model <model_name> \
    --api_key <your_openai_api_key> \
    --temperature <temperature> \
```

Even if you are generating the factoid answers for the dataset itself, make sure to follow the --data path format to ensure all the parameters are set correctly.
Factoid answer generation using 4 (top_k > 0: with context | top_k = 0: without context) different prompts (metrics are either 0:(EM, ROUGE-F1, Semantic Similarity) or 1:(Sacc, Lacc)):
```
python factoid_answer_generator.py \
      --data results/<dataset_name>/retrieved_passages_<model_name>.json \
      --model <model_name> \
      --top_k <0,1,5,10> \
      --api_key <your_openai_api_key> \
      --metrics <0,1>
```
