# GReG: A Training-free Generate-Retrieve-Generate Pipeline

<table style="border: none;">
  <tr>
    <td valign="top" width="80%">
      <h2>From Hints to Answers: Uncertainty-Aware LLM-Guided Retrieval for Multi-Hop Question Answering</h2>
      <p>We propose a <b>G</b>enerate-<b>Re</b>trieve-<b>G</b>enerate (GReG) pipeline for multi-hop open-domain question answering (QA) that improves the accuracy of Retrieval-Augmented Language Models (RALMs). RALMs improve answer generation by incorporating external evidence into the model's input. The proposed GReG pipeline leverages the parametric knowledge of large language models (LLMs) to produce semantically rich “hints” in the form of long-form answers, which are repurposed as retrieval queries to obtain relevant supporting evidence. To mitigate hallucination and improve reliability, GReG includes an entropy-based uncertainty estimation method that filters out low-confidence hints. Overall, GReG enables a smaller, more efficient language model to answer complex questions more accurately by augmenting it with high-quality evidence retrieved through hint-based search. Experiments on prominent multi-hop QA datasets show that GReG outperforms strong baselines under identical retrieval and generation settings.</p>
    </td>
    <td valign="top" width="20%">
      <img src="img/greg.png" width="100%">
    </td>
  </tr>
</table>

### Initial Setup:
```
conda create --name greg python=3.12
conda activate greg
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
git clone https://github.com/Bitazad/GReG.git
cd GReG
bash download_datasets.sh # requires the subsample.py file
bash run_hints_batch.sh # requires the hint_generator.py file
```

Note that you can change the model name or set _using_batch_api_ to True to obtain results instantly. Otherwise, you'll need to run the next command.
```
bash run_hints_prev.sh
```
Make sure to insert your OpenAI API key.
Also, ensure that all batch runs have completed before executing this command by checking your OpenAI Batch API dashboard.

Now run the following command to process the folds and select the final retrieval query based on the median token entropy:
```
python3 hint_selector.py
```
This script also saves a histogram showing the distribution of folds that contributed to selecting the best query.

Now it's time to retrieve relevant Wikipedia passages using the best (most certain) queries saved in _runs/hints/{dataset}/{model}/best_queries.jsonl. This retrieval is performed using ColBERTv2 and the Wikipedia 2017 abstracts dump.
```
python3 retriever.py
```

## Prompts
We use the following prompts to obtain results from various LLMs:

### Hint Generation (New retrieval query)
```
system_instruction = "When answering questions, always include relevant information from the question in your response."
```
We provide only the question to the model to generate a long-form, hint-enriched answer. This step applies exclusively to the GPT-based models used in this paper for hint generation; namely, GPT-4o and GPT-3.5-turbo-instruct.

### Answer Generation (Factoid answers)
- GPT Models:
  - top_k > 0 (include augmented context)
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
  - top_k = 0 (No context augmentation)
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
    f"System: This is a chat between a user and an artificial intelligence assistant\n\nUser:{initial_prompt}\n<Question>:{datum['question_org']}\n\nAssistant:\nSo, the answer is:"
)
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
conversation = f"User:{instruction}\n<Question>: {question}\n\n"
formatted_input = f"{system}\n\n{conversation}Assistant:\nSo, the answer is:"
```
