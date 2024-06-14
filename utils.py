from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import numpy as np
import spacy
import re

def map_entropy_to_color(entropy, min_entropy, max_entropy, inv):
    if max_entropy != min_entropy:
        normalized_entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
    else:
        normalized_entropy = entropy
    colormap = plt.get_cmap('coolwarm') if inv==False else plt.get_cmap('coolwarm_r')
    rgba_color = colormap(normalized_entropy)
    hex_color = mcolors.to_hex(rgba_color)
    return hex_color

def colorize_tokens(tokens, entropies, threshold=None, inv=False, placeholder=True):
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    colorized_tokens = []
    for token, entropy in zip(tokens, entropies):
        color = map_entropy_to_color(entropy, min_entropy, max_entropy, inv)
        if threshold == None or entropy < threshold:
            colorized_tokens.append(f'<span style="color:{color}">{token}</span>')
        else:
            if placeholder == False:
                colorized_tokens.append(f'<span style="color:{color}">{token}</span>')
            else:
                colorized_tokens.append(f'<span style="color:{color}">_</span>')
    return ' '.join(colorized_tokens)

def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:  # To avoid log(0)
            entropy -= p * math.log2(p)
    return entropy

def entropy_for_tokens(token_probabilities):
    entropies = []
    for probabilities in token_probabilities:
        entropy = calculate_entropy(probabilities)
        entropies.append(entropy)
    return entropies

def apply_prev_word_entropies(words_entropies):
    entropies = []
    before = 0
    prev = 0
    for entropy in words_entropies:
        entropies.append(entropy * np.exp(prev) * np.exp(before))
        before = prev
        prev = entropy
    return entropies

def apply_prev_word_entropies_geo(words_entropies):
    entropies = []
    before = 0
    prev = 0
    for entropy in words_entropies:
        entropies.append(np.sqrt(entropy * prev * before))
        before = prev
        prev = entropy
    return entropies

def apply_window_word_entropies(words_entropies):
    entropies = []
    before = 0
    for i in range(len(words_entropies)):
        entropy = words_entropies[i]
        if i == len(words_entropies) - 1:
            break
        after = words_entropies[i + 1]
        entropies.append(entropy * np.exp(before) * np.exp(after))
        before = entropy
    entropies.append(words_entropies[-1] * np.exp(words_entropies[-2]))
    return entropies

def get_word_entropies(tokens, token_entropies):
  def combine_entropies(entropies):
    return max(entropies)
  words = []
  word_entropies = []
  current_word = ""
  current_entropies = []

  for token, entropy in zip(tokens, token_entropies):
    if token.startswith(' ') and current_word:
      words.append(current_word)
      word_entropies.append(combine_entropies(current_entropies))
      current_word = token.strip()
      current_entropies = [entropy]
    else:
      current_word += token
      current_entropies.append(entropy)

  if current_word:
    words.append(current_word)
    word_entropies.append(combine_entropies(current_entropies))

  return words, word_entropies

def normalize_string(input_string):
  pattern = r'[.,;:!?]'
  normalized_string = re.sub(pattern, '', input_string)
  return normalized_string

def get_entities(sentence):
  entities = []
  nlp = spacy.load("en_core_web_md")
  doc = nlp(sentence)
  for ent in doc.ents:
    entities.append(ent.text)
  return entities

def filter_words(words, word_entropies, question_entities, output_entities, threshold, ignore_entropies=False, placeholder=' '):
  if threshold == 'median' or threshold == 'p-50':
    th = np.median(word_entropies)
  elif threshold == 'p-25':
    th = np.percentile(word_entropies, 25)
  else:
    th = None
  filtered_words = []
  for word, entropy in zip(words, word_entropies):
    norm_word = normalize_string(word)
    if ignore_entropies == False:
      if entropy > th:
        if not (any(norm_word in ent for ent in question_entities)): # word is not an entity within the question
          if any(norm_word in ent for ent in output_entities): # word is an entity
            filtered_words.append(placeholder)
            continue
      filtered_words.append(word)
    else:
      if not (any(norm_word in ent for ent in question_entities)): # word is not an entity within the question
        if any(norm_word in ent for ent in output_entities): # word is an entity
          filtered_words.append(placeholder)
          continue
      filtered_words.append(word)
  return filtered_words

def generate_ngrams(words, n):
    words = list(map(normalize_string, words))
    ngrams_with_indices = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        indices = list(range(i, i+n))
        ngrams_with_indices.append((ngram, indices))
    return ngrams_with_indices

def locate_ngrams(words, n, output_entities):
    sentence_ngrams_with_indices = generate_ngrams(words, n)
    found_ngrams_with_indices = [(ngram, indices) for ngram, indices in sentence_ngrams_with_indices if ngram in list(map(normalize_string, output_entities))]
    return found_ngrams_with_indices

def combine_dependent_words_entropies(words, word_entropies, output_entities, max_window_size=5):
  for n in range(1, max_window_size + 1):
    found_ngrams = locate_ngrams(words, n, output_entities)
    for ngram in found_ngrams:
      ents = []
      for i in ngram[1]:
        ents.append(word_entropies[i])
      max_ents = max(ents)
      for i in ngram[1]:
        word_entropies[i] = max_ents
  return word_entropies
