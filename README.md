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
conda create --name geregen python=3.12
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

### Prompts
We leverage the following prompts to get the results from different LLMs:

#### Hint Generation (new retrieval query)
```
system_instruction = "When answering questions, always include relevant information from the question in your response."
```
and we only provide the model with the question to get a long-form (hint-enriched) answer. This is only for GPT based models used in this paper for hint generation (GPT-4o and GPT-3.5-turbo-instruct).

#### Answer Generation (factoid answers)
- GPT Models:
  - top_k > 0 (have augmented context)
```
system = "System: This is a chat between a user and an artificial intelligence assistant."
initial_prompt = (
                "You are an expert assistant in answering complex and **Multi-Hop** questions after <Question>. "
                "Your task is to provide a concise, factoid-style answer based strictly on the information enclosed within <doc> and </doc> tags.\n\n"

                "**Instructions:**\n"
                "1. Use logical reasoning *only* if the document lacks direct information.\n"
                "2. DO NOT include your thought process or explanation in the final output.\n\n"

                "**Formatting Rules:**\n"
                "- If the question is yes/no, answer with **yes** or **no** only.\n"
                "- If the answer is a date or place, give only the date or place.\n"
                "- If the answer is not clearly stated, use reasoning but respond in a maximum of 2–3 words.\n"
                "- If you DO NOT know the answer, DO NOT generate anything.\n\n"

                "**Examples:**\n\n"

                "Example 1:\n"
                "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
                "So, the answer is: Little Richard\n"

                "Example 2:\n"
                "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
                "So, the answer is: Chinua Achebe\n"

                "Example 3:\n"
                "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
                "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
                "So, the answer is: 1979\n\n"
)
full_prompt = (
                    f"{system}\n\n"
                    f"User:{initial_prompt}\n"
                    f"Now use the given knowledge below to answer the question. Internally reason step-by-step, but Output only the final answer, nothing else.\n\n" 
                    f"<doc>\n{doc_content}\n</doc>\n"
                    f"<Question>: {datum['question']}\nAssistant:\nSo, the answer is:"
                )
```
  - top_k = 0 (no context augmentation)
```
initial_prompt = (   
                    "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>."
                    "You must use logical reasoning to arrive at the best possible answer.\n"
                    "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
                    "In case of referring to a date or specific place, just name the date or place.\n"
                    "The answer must be concise (2–3 words max), factoid-style.\n"  

                    "Here are some examples:\n\n"

                    "Example 1:\n"
                    "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
                    "So, the answer is: Little Richard\n\n"

                    "Example 2:\n"
                    "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
                    "So, the answer is: Chinua Achebe\n\n"

                    "Example 3:\n"
                    "<Question>: 'Remember Me Ballin’' is a CD single by Indo G that features an American rapper born in what year?\n"
                    "So, the answer is: 1979\n\n"
                )

full_prompt = (
                f"System: This is a chat between a user and an artificial intelligence assistant\n\nUser:{initial_prompt}\n<Question>:{datum['question_org']}\n\nAssistant:\nSo, the answer is:")
```
- Others (Vicuna and Qwen):
```
  system = "System: This is a chat between a user and an artificial intelligence assistant."
```
  - top_k > 0
```
    instruction = (
            "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question> based on the provided knowledge enclosed within <doc> and </doc> tags.\n"
            "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
            "In case of referring to a date or specific place, just name the date or place.\n"
            "If the knowledge contains the answer, give a **concise, factoid-style answer (2–3 words max)**.\n"
            "**IMPORTANT: Do NOT include any thought process, explanation, or reasoning. Only return the final answer after <Answer>.**\n"

            "Here are some examples:\n\n"

            "Example 1:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
            "So, the answer is: Little Richard\n\n"

            "Example 2:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
            "So, the answer is: Chinua Achebe\n\n"

            "Example 3:\n"
            "<doc>\n{{KNOWLEDGE FOR YOUR REFERENCE}}\n</doc>\n"
            "<Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?\n"
            "So, the answer is: 1979\n\n"
        )

        all_contexts = ""
        for j in range(top_k):
            title = contexts[j]['title']
            text = contexts[j]['text']
            doc_text = f"Title: {title}\nText: {text}" if include_titles else text
            all_contexts += f"{doc_text}\n\n"

        doc_section = f"<doc>\n{all_contexts.strip()}\n</doc>"
        formatted_input = (
            f"{system}\n\n"
            f"User:{instruction}\n\n"
            f"Now use the given knowledge below to answer the question.\n\n"
            f"{doc_section}\n\n"
            f"<Question>: {question}\nAssistant:\nSo, the answer is:"
        )
        print(formatted_input)
```

  - top_k = 0

```
      instruction = (
            "As an expert assistant in answering complex and **Multi-Hop** questions, your task is to answer the given question after <Question>."
            "In case of **yes/no** questions, **only** answer with **yes** or **no**.\n"
            "In case of referring to a date or specific place, just name the date or place.\n"
            "The answer must be concise (2–3 words max), factoid-style.\n"  
            "Here are some examples:\n\n"

            "Example 1:\n"
            "<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and was born on December 5, 1932?\n"
            "So, the answer is: Little Richard\n\n"

            "Example 2:\n"
            "<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?\n"
            "So, the answer is: Chinua Achebe\n\n"

            "Example 3:\n"
            "<Question>: 'Remember Me Ballin’' is a CD single by Indo G that features an American rapper born in what year?\n"
            "So, the answer is: 1979\n\n"
        )
```

        conversation = f"User:{instruction}\n<Question>: {question}\n\n"
        formatted_input = f"{system}\n\n{conversation}Assistant:\nSo, the answer is:"
