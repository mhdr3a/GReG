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

### Initial Setup:
```
conda create --name geregen
conda activate geregen

conda install -c conda-forge unidecode
pip install openai
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge spacy
pip install transformers
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge termcolor
pip install dspy
python3 -m spacy download en_core_web_md
git clone https://github.com/mhdr3a/GReG.git
cd GReG
bash download_datasets.sh # requires the subsample.py file
bash run_hints_batch.sh # requires the hint_generator.py file .. Remember to wipe off your open ai api key
```

Note that you can change the model name, or the using_batch_api option (to get the results instantaneously), otherwise, you'll need to run the next command
```
bash run_hints_prev.sh
```
Remember to insert your open ai api key
Note that you need to make sure that all the batch runs are done before running this bash command by checking your open ai batch api dashboard

Now run the following command to process the folds and select the final retrieval query based on the median value of token entropies.
```
python3 hint_selector.py
```
it also saves a histogram of the distribution of the folds candidating the best query in the final selection of queries.

now it's time for retrieval of relevant Wikipedia passages using the best queries (most certain queries) saved in runs/hints/{dataset}/{model}/best_queries.jsonl using ColBERTv2 and from Wikipedia 2017 abstracts dump.

```
python3 retriever.py
```

