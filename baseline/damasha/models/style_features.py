import torch
import torch.nn as nn
import numpy as np
from nltk import pos_tag
import textstat

class StyleFeatureExtractor:
    """
    Extracts style-based features from text for use in the model.
    """
    def __init__(self, tokenizer, language_model, device):
        self.tokenizer = tokenizer
        self.lm = language_model
        self.device = device
        self.pos_tags = set(['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'])
        self.lm_classifier = nn.Linear(self.lm.config.hidden_size, self.tokenizer.vocab_size).to(device)

    def compute_lexical_diversity(self, words, word_ids, window=5):
        ttr_scores = []
        for i in range(len(words)):
            start = max(0, i - window // 2)
            end = min(len(words), i + window // 2 + 1)
            window_words = words[start:end]
            types = len(set(window_words))
            tokens = len(window_words)
            ttr = types / tokens if tokens > 0 else 0
            ttr_scores.append(ttr)
        return [ttr_scores[word_id] if word_id is not None and word_id < len(ttr_scores) else 0.0 for word_id in word_ids]

    def compute_punctuation_density(self, words, word_ids):
        punc_chars = set('.,!?;:"\'')
        punc_density = [sum(1 for c in word if c in punc_chars) / len(word) if len(word) > 0 else 0 for word in words]
        return [punc_density[word_id] if word_id is not None and word_id < len(punc_density) else 0.0 for word_id in word_ids]

    def compute_pos_density(self, words, word_ids):
        pos_tags = pos_tag(words)
        pos_density = [1 if tag in self.pos_tags else 0 for _, tag in pos_tags]
        return [pos_density[word_id] if word_id is not None and word_id < len(pos_density) else 0.0 for word_id in word_ids]

    def compute_readability(self, words, word_ids):
        text = ' '.join(words)
        score = textstat.flesch_kincaid_grade(text) if text.strip() else 0
        readability = [score / 20.0] * len(words)
        return [readability[word_id] if word_id is not None and word_id < len(readability) else 0.0 for word_id in word_ids]

    def get_style_features(self, words, word_ids):
        ttr = self.compute_lexical_diversity(words, word_ids)
        punc = self.compute_punctuation_density(words, word_ids)
        pos = self.compute_pos_density(words, word_ids)
        read = self.compute_readability(words, word_ids)
        return torch.tensor(np.stack([ttr, punc, pos, read], axis=-1), dtype=torch.float).to(self.device)