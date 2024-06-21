# GReG

<table style="border: none;">
  <tr>
    <td valign="top" width="80%" style="border-width: 0 !important;">
      <h2>Generate-Retrieve-Generate Framework for Open Domain Question Answering</h2>
      <p><b>G</b>enerate-<b>Re</b>trieve-<b>G</b>enerate (GReG) Framework for Open Domain Question Answering is a sophisticated AI architecture designed to tackle complex queries that require synthesizing information from multiple sources or steps. This technology stands at the cutting edge of natural language processing and machine learning, offering significant improvements over traditional single-passage retrieval systems.</p>
    </td>
    <td valign="top" width="20%" style="border-width: 0 !important;">
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
cd ..
git clone https://github.com/mhdr3a/GReG.git
cd GReG
bash download_datasets.sh # MuSiQue, HotpotQA, IIRC, and 2WikiMultihopQA
```

Replace ```FiD/src/data.py``` with ```GReG/src/data.py```

Replace ```FiD/src/evaluation.py``` with ```GReG/src/evaluation.py```

```
python ../FiD/passage_retrieval.py \
    --model_path ../FiD/pretrained_models/nq_retriever \
    --passages ../FiD/open_domain_data/psgs_w100.tsv \
    --data <dataset_dir> \
    --passages_embeddings ../FiD/wikipedia_embeddings_00 \
    --output_path <output_dir> \
    --n-docs 10 \
```
