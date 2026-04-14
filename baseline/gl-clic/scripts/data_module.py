import os
import sys
import math
import h5py
import json
import spacy
import torch
import random
import joblib
import pickle
import logging
import numpy as np
import pandas as pd
import lightning as L
import multiprocessing

from tqdm import tqdm
from cefrpy import CEFRSpaCyAnalyzer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.config import GLCLiCModelConfig
from torch.utils.data import DataLoader
from scripts.dataset import GLCliCCoAuthorDataset, GLCliCSeqXGPTBenchDataset
from utils.parse_coauthor_dataset import parse_data as coauthor_parse_data
from utils.parse_seqxgpt_bench_dataset import parse_data as seqxgpt_bench_parse_data


nlp = spacy.load("en_core_web_sm")


_worker_analyzer = None
_worker_helper_instance = None

def _preprocess_worker(data_item, abbreviation_mapping, tokenizer_config):
    global _worker_analyzer
    global _worker_helper_instance

    if _worker_analyzer is None:
        _worker_analyzer = RobustCEFRSpaCyAnalyzer(abbreviation_mapping=abbreviation_mapping)

    if _worker_helper_instance is None:
        _worker_helper_instance = GLCliCSeqXGPTBenchDataModule.for_worker(tokenizer_config, abbreviation_mapping)

    return _worker_helper_instance.single_preprocess(data_item, _worker_analyzer)



class GLCliCCoAuthorDataModule(L.LightningDataModule):

    def __init__(self,
                 model_config: GLCLiCModelConfig,
                 raw_data_path: str,
                 parsed_data_path: str,
                 preprocessed_data_path: str,
                 essay_type: str,
                 batch_size: int,
                 recreate: bool = False,
                 test: bool = False):

        super(GLCliCCoAuthorDataModule, self).__init__()

        assert essay_type in ["creative", "argumentative", "all"]

        self.model_config = model_config
        self.raw_data_path = raw_data_path
        self.parsed_data_path = parsed_data_path
        self.preprocessed_data_path = preprocessed_data_path

        self.essay_type = essay_type
        self.batch_size = batch_size
        self.recreate = recreate
        self.test = test

        additional_special_tokens = ["[GLOBAL COHERENCE]", "[LOCAL COHERENCE]", "[GLOBAL LEXICAL]", "[LOCAL LEXICAL]", "[REPRESENTATION]"]

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        self.local_lexical_special_token_id = self.tokenizer.convert_tokens_to_ids("[LOCAL LEXICAL]")
        self.max_len = 512

        self.abbreviation_mapping = {
            "'m": "am",
            "'s": "is",
            "'re": "are",
            "'ve": "have",
            "'d": "had",
            "n't": "not",
            "'ll": "will"
        }

        self.label_mapping = {
            0: "Human",
            1: "AI",
            2: "Human-AI"
        }

        self.type_mapping = {
            "creative": 0,
            "argumentative": 1
        }

        if not os.path.exists(raw_data_path):
            raise FileNotFoundError('[DATA] Raw data not found, please download the data at https://github.com/douglashiwo/AISentenceDetection')

        self.parse_raw_data()

        self.raw_data = pickle.load(open(self.parsed_data_path, 'rb'))

        if recreate or not os.path.exists(self.preprocessed_data_path):
            print("\n[DATA] Preprocessing data...\n")
            self.data = self.preprocess()
        else:
            print("\n[DATA] Loading preprocessed data...\n")
            self.data = pickle.load(open(self.preprocessed_data_path, 'rb'))

            # If the data is less than 3000, preprocess it again (this is because of the test mode)
            if len(self.data) < 3000:
                print("\n[DATA] Preprocessing data again due to test mode...\n")
                self.data = self.preprocess()

        if self.test:
            print(f"\n[DATA] Limit data to maks \"50\" in \"Test Mode\"")
            self.data = self.data[:50]


    def parse_raw_data(self):
        print()

        print("\n[DATA] Processed data not found or recreate = True, parsing raw data...\n")
        os.makedirs(os.path.dirname(self.parsed_data_path), exist_ok=True)

        print("\n[DATA] Parsing raw data...\n")
        coauthor_parse_data(self.raw_data_path, self.parsed_data_path)

        print("\n[DATA] Done parsing data\n")
        print()


    def filter_data(self, data, stage, type):
        filtered_data = []

        for d in data:
            if stage == "test":
                if d['stage'] == stage:
                    filtered_data.append(d)
                    continue

            if type == "all":
                if d['stage'] == stage:
                    filtered_data.append(d)
                    continue

            if d['type'] == type and d["stage"] == stage:
                filtered_data.append(d)
                continue

        return filtered_data


    def generate_incoherent_doc(self, sentences, total):
        max_tries = 100

        incoherent_docs_sentence_index = []
        coherent_doc_sentence_index = list(range(len(sentences)))

        for i in range(total):
            tries = 0

            while tries < max_tries:
                incoherent_doc_sentence_index = coherent_doc_sentence_index.copy()
                random.shuffle(incoherent_doc_sentence_index)

                if incoherent_doc_sentence_index != coherent_doc_sentence_index:
                    if incoherent_doc_sentence_index not in incoherent_docs_sentence_index:
                        incoherent_docs_sentence_index.append(incoherent_doc_sentence_index)
                        break

                tries += 1

        return coherent_doc_sentence_index, incoherent_docs_sentence_index


    def generate_incoherent_triplet(self, sentences, target_center_sentence_index, total):
        max_tries = 100

        incoherent_triplets = []
        coherent_triplet = [target_center_sentence_index]

        left_index = target_center_sentence_index - 1
        right_index = target_center_sentence_index + 1

        if left_index >= 0:
            coherent_triplet.insert(0, left_index)

        if right_index < len(sentences):
            coherent_triplet.append(right_index)

        for i in range(total):
            tries = 0

            while tries < max_tries:
                incoherent_triplet = [target_center_sentence_index]

                random_left_index = random.randint(0, len(sentences) - 1)
                random_right_index = random.randint(0, len(sentences) - 1)

                if target_center_sentence_index != 0:
                    if random_left_index != target_center_sentence_index:
                        if abs(target_center_sentence_index - random_left_index) > 1:
                            incoherent_triplet.insert(0, random_left_index)

                if target_center_sentence_index != len(sentences) - 1:
                    if random_right_index != target_center_sentence_index:
                        if abs(random_right_index - target_center_sentence_index) > 1:
                            incoherent_triplet.append(random_right_index)

                if incoherent_triplet != coherent_triplet:
                    if incoherent_triplet not in incoherent_triplets:
                        if len(incoherent_triplet) == len(coherent_triplet):
                            incoherent_triplets.append(incoherent_triplet)
                            break

                tries += 1

        if len(incoherent_triplets) < total:
            print("Warning: Not enough incoherent triplets generated")
            print(target_center_sentence_index)
            print(incoherent_triplets)

        return coherent_triplet, incoherent_triplets


    def truncate_doc(self, sentence_index, sentences_token_length, target_sentence_index):
        doc = []
        total_token_length = 0

        aligned_target_sentence_index = sentence_index.index(target_sentence_index)

        doc.append(target_sentence_index)
        total_token_length += sentences_token_length[aligned_target_sentence_index]


        for i in range(len(sentence_index)):
            if i == 0:
                continue

            left_index = aligned_target_sentence_index - i
            right_index = aligned_target_sentence_index + i

            if left_index >= 0:

                if total_token_length + sentences_token_length[left_index] > self.max_len:
                    break

                doc.insert(0, sentence_index[left_index])
                total_token_length += sentences_token_length[left_index]

            if right_index < len(sentence_index):

                if total_token_length + sentences_token_length[right_index] > self.max_len:
                    break

                doc.append(sentence_index[right_index])
                total_token_length += sentences_token_length[right_index]

        return doc


    def build_doc(self, sentences, sentence_index, target_sentence_index):
        sentences_copy = sentences.copy()
        sentences_copy[target_sentence_index] = f"{self.tokenizer.sep_token}{sentences_copy[target_sentence_index]}{self.tokenizer.sep_token}"
        doc = " ".join([sentences_copy[i] for i in sentence_index])
        return doc


    def build_triplet(self, sentences, triplet, target_sentence_index):
        sentences_copy = sentences.copy()
        sentences_copy[target_sentence_index] = f"{self.tokenizer.sep_token}{sentences_copy[target_sentence_index]}{self.tokenizer.sep_token}"

        triplet_doc = []
        for i in triplet:
            triplet_doc.append(sentences_copy[i])
        triplet_doc = " ".join(triplet_doc)

        return triplet_doc


    def encode_input(self, input):
        encoded_input = self.tokenizer(input, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, padding_side="right", padding='max_length', truncation=True)

        input_ids = encoded_input['input_ids'][0]
        attention_mask = encoded_input['attention_mask'][0]
        token_type_ids = encoded_input['token_type_ids'][0]

        return torch.stack([input_ids, attention_mask, token_type_ids])


    def encode_inputs(self, inputs):
        encoded_input = self.tokenizer(inputs, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, padding_side="right", padding='max_length', truncation=True)

        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        token_type_ids = encoded_input['token_type_ids']

        return torch.stack([input_ids, attention_mask, token_type_ids])


    def encode_cefr_representation(self, sentences, truncated_doc):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        sentence_idxs = []

        sep_token = self.tokenizer.sep_token

        for sentence_index in truncated_doc:
            sentence = sentences[sentence_index]
            encoded_input = self.tokenizer(sentence, add_special_tokens=False)

            input_ids.extend(encoded_input['input_ids'])
            attention_mask.extend(encoded_input['attention_mask'])
            token_type_ids.extend(encoded_input['token_type_ids'])
            sentence_idxs.extend([sentence_index] * len(encoded_input['input_ids']))

        max_len_wo_special_tokens = self.max_len - 3

        if len(input_ids) > max_len_wo_special_tokens:
            input_ids = input_ids[:max_len_wo_special_tokens]
            attention_mask = attention_mask[:max_len_wo_special_tokens]
            token_type_ids = token_type_ids[:max_len_wo_special_tokens]
            sentence_idxs = sentence_idxs[:max_len_wo_special_tokens]

        # add special tokens
        input_ids = [self.tokenizer.cls_token_id] + [self.local_lexical_special_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] + [1] + attention_mask + [1]
        token_type_ids = [0] + [0] + token_type_ids + [0]
        sentence_idxs = [-1] + [-1] + sentence_idxs + [-1]

        if len(input_ids) < self.max_len:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            attention_mask += [0] * (self.max_len - len(attention_mask))
            token_type_ids += [0] * (self.max_len - len(token_type_ids))
            sentence_idxs += [-1] * (self.max_len - len(sentence_idxs))

        return torch.stack([torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(sentence_idxs)])


    def truncate_global_cefr(self, word_cefr_levels, word_sentence_idx, focus_sentence_idx):

        if focus_sentence_idx not in word_sentence_idx:
            offet = 1
            while focus_sentence_idx not in word_sentence_idx:
                new_focus_sentence_idx = focus_sentence_idx + offet
                if new_focus_sentence_idx in word_sentence_idx:
                    focus_sentence_idx = new_focus_sentence_idx
                    break

                new_focus_sentence_idx = focus_sentence_idx - offet
                if new_focus_sentence_idx in word_sentence_idx:
                    focus_sentence_idx = new_focus_sentence_idx
                    break

                offet += 1


        focus_indices = [i for i, sentence_id in enumerate(word_sentence_idx)
                        if sentence_id == focus_sentence_idx]

        center_focus_idx = focus_indices[len(focus_indices) // 2]

        num_left = math.floor((self.max_len - 1) / 2)
        num_right = math.ceil((self.max_len - 1) / 2)

        ideal_start_idx = center_focus_idx - num_left
        ideal_end_idx = center_focus_idx + num_right

        list_len = len(word_sentence_idx)
        actual_start_idx = ideal_start_idx
        actual_end_idx = ideal_end_idx # Inclusive

        if actual_start_idx < 0:
            shift = 0 - actual_start_idx
            actual_start_idx = 0
            actual_end_idx += shift

        if actual_end_idx >= list_len:
            shift = actual_end_idx - (list_len - 1) # Amount we went past the end
            actual_end_idx = list_len - 1 # Clamp to the last valid index
            actual_start_idx -= shift # Shift the start index further left

        actual_start_idx = max(0, actual_start_idx)

        actual_end_slice = min(list_len, actual_end_idx + 1)

        if actual_end_slice - actual_start_idx < self.max_len:
            actual_start_idx = max(0, actual_end_slice - self.max_len)

        result_indices = list(range(actual_start_idx, actual_end_slice))

        final_word_cefr_levels = [value for i, value in enumerate(word_cefr_levels) if i in result_indices]

        if len(final_word_cefr_levels) > self.max_len:
            final_word_cefr_levels = final_word_cefr_levels[:self.max_len]
        elif len(final_word_cefr_levels) < self.max_len:
            final_word_cefr_levels += [-1] * (self.max_len - len(final_word_cefr_levels))

        return final_word_cefr_levels


    def preprocess(self):

        preprocessed_data = []
        cefr_analyzer = CEFRSpaCyAnalyzer(abbreviation_mapping=self.abbreviation_mapping)

        raw_data = self.raw_data[:30] if self.test else self.raw_data

        for data in tqdm(raw_data):
            stage = data['train_ix']
            type = data['type']
            labels = data['labels']

            sentences = data['sentences']
            sentences_token_length = [len(self.tokenizer.tokenize(sentence)) for sentence in sentences]

            coherencent_doc_sentence_index, incoherent_docs_sentence_index = self.generate_incoherent_doc(sentences, 3)

            ### Lexical ###
            sentence_cefr_levels = []
            word_cefr_levels = []
            word_sentence_idx = []

            for sentence_idx, sentence in enumerate(sentences):
                tokens = cefr_analyzer.analize_doc(nlp(sentence))
                cefr_data = []
                for token in tokens:
                    _, _, is_skipped, level, _, _ = token
                    if not is_skipped:
                        cefr_data.append(level)
                        word_cefr_levels.append(level)
                        word_sentence_idx.append(sentence_idx)

                sentence_cefr_level = (sum(cefr_data) / len(cefr_data)) if len(cefr_data) > 0 else 1
                sentence_cefr_levels.append(sentence_cefr_level)

            if len(sentence_cefr_levels) < 128:
                sentence_cefr_levels += [-1] * (128 - len(sentence_cefr_levels))

            global_cefr_level = (sum(word_cefr_levels) / len(word_cefr_levels)) if len(word_cefr_levels) > 0 else 1

            #### Coherence ####
            for i in range(len(sentences)):

                # Global Coherence
                truncated_coherent_doc = self.truncate_doc(coherencent_doc_sentence_index, sentences_token_length, i)
                truncated_incoherent_docs = [self.truncate_doc(incoherent_docs_sentence_index[j], sentences_token_length, i) for j in range(3)]

                coherent_doc = "[GLOBAL COHERENCE]" + self.build_doc(sentences, truncated_coherent_doc, i)
                incoherent_docs = ["[GLOBAL COHERENCE]" + self.build_doc(sentences, truncated_incoherent_docs[j], i) for j in range(3)]

                # Local Coherence
                coherent_triplet, incoherent_triplets = self.generate_incoherent_triplet(sentences, i, 3)
                coherent_triplet = "[LOCAL COHERENCE]" + self.build_triplet(sentences, coherent_triplet, i)
                incoherent_triplets = ["[LOCAL COHERENCE]" + self.build_triplet(sentences, incoherent_triplets[j], i) for j in range(3)]

                # Global Lexical
                global_cefr_levels = self.truncate_global_cefr(word_cefr_levels, word_sentence_idx, i)

                # Local Lexical
                local_cefr_levels = [sentence_idx for sentence_idx, x in enumerate(sentence_cefr_levels) if round(x) == round(sentence_cefr_levels[i])]
                local_cefr_levels += [-1] * (128 - len(local_cefr_levels))

                label = labels[i]
                sentence = "[REPRESENTATION]" + sentences[i]

                #### Encode Inputs ####
                coherent_doc_input = self.encode_input(coherent_doc)
                incoherent_docs_input = self.encode_inputs(incoherent_docs)

                coherent_triplet_input = self.encode_input(coherent_triplet)
                incoherent_triplets_input = self.encode_inputs(incoherent_triplets)

                global_cefr_levels = torch.tensor(global_cefr_levels)
                local_cefr_levels = torch.tensor(local_cefr_levels)

                cefr_input = self.encode_cefr_representation(sentences, truncated_coherent_doc)

                sentence_input = self.encode_input(sentence)

                label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=3)

                preprocessed_data.append({
                    'stage': stage,
                    'label': label,
                    'type': type,
                    'coherent_doc_input': coherent_doc_input,
                    'incoherent_docs_input': incoherent_docs_input,
                    'coherent_triplet_input': coherent_triplet_input,
                    'incoherent_triplets_input': incoherent_triplets_input,
                    'global_cefr_levels': global_cefr_levels,
                    'global_cefr_level': global_cefr_level,
                    'local_cefr_levels': local_cefr_levels,
                    'local_cefr_level': sentence_cefr_levels[i],
                    'cefr_input': cefr_input,
                    'sentence_input': sentence_input,
                })

        os.makedirs(os.path.dirname(self.preprocessed_data_path), exist_ok=True)

        pickle.dump(preprocessed_data, open(self.preprocessed_data_path, 'wb'))

        return preprocessed_data


    def setup(self, stage=None):
        if stage=='fit':
            self.train_filtered = self.filter_data(self.data, "train", self.type)
            self.val_filtered = self.filter_data(self.data, "valid", self.type)

            self.train_dataset = GLCliCCoAuthorDataset(self.train_filtered, self.model_config)
            self.val_dataset = GLCliCCoAuthorDataset(self.val_filtered, self.model_config)
        elif stage=='test':
            self.test_filtered = self.filter_data(self.data, "test", self.type)
            self.test_dataset = GLCliCCoAuthorDataset(self.test_filtered, self.model_config)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())



class GLCliCSeqXGPTBenchDataModule(L.LightningDataModule):

    @classmethod
    def for_worker(cls, tokenizer_config, abbreviation_mapping):
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config['name'])
        instance.tokenizer.add_special_tokens({'additional_special_tokens': tokenizer_config['special_tokens']})
        instance.max_len = 512
        instance.label_mapping = {"U": 0, "A": 1}
        instance.abbreviation_mapping = abbreviation_mapping
        instance.local_lexical_special_token_id = instance.tokenizer.convert_tokens_to_ids("[LOCAL LEXICAL]")
        return instance

    def __init__(self,
                 model_config: GLCLiCModelConfig,
                 raw_data_path: str,
                 parsed_data_path: str,
                 preprocessed_data_path: str,
                 batch_size: int,
                 recreate: bool = False,
                 test: bool = False):

        super(GLCliCSeqXGPTBenchDataModule, self).__init__()

        self.model_config = model_config
        self.raw_data_path = raw_data_path
        self.parsed_data_path = parsed_data_path
        self.preprocessed_data_path = preprocessed_data_path

        self.batch_size = batch_size
        self.recreate = recreate
        self.test = test

        raw_files = [file_name for file_name in os.listdir(raw_data_path) if file_name.endswith('.jsonl')]

        if len(raw_files) == 0:
            raise FileNotFoundError('[DATA] Raw data not found, please download the data at https://github.com/Jihuai-wpy/SeqXGPT/tree/main/SeqXGPT/dataset/SeqXGPT-Bench')

        if not os.path.exists(parsed_data_path) or self.recreate:
            self.parse_raw_data()

        self.train_sources = model_config.train_sources
        self.test_sources = model_config.test_sources

        additional_special_tokens = ["[GLOBAL COHERENCE]", "[LOCAL COHERENCE]", "[GLOBAL LEXICAL]", "[LOCAL LEXICAL]", "[REPRESENTATION]"]

        self.tokenizer_config = {
            'name': 'microsoft/deberta-v3-base',
            'special_tokens': additional_special_tokens
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_config['name'])
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.tokenizer_config['special_tokens']})
        self.max_len = 512

        self.local_lexical_special_token_id = self.tokenizer.convert_tokens_to_ids("[LOCAL LEXICAL]")

        self.abbreviation_mapping = {
            "'m": "am",
            "'s": "is",
            "'re": "are",
            "'ve": "have",
            "'d": "had",
            "n't": "not",
            "'ll": "will"
        }

        self.label_mapping = {
            "U": 0, # Human
            "A": 1, # AI
        }

        self.source_mapping = {
            "gpt2": 0,
            "gpt3": 1,
            "gptj": 2,
            "gptneo": 3,
            "human": 4,
            "llama": 5,
        }

        self.index_map = []

        self.raw_data = pickle.load(open(self.parsed_data_path, 'rb'))

        if recreate or not os.path.exists(self.preprocessed_data_path):
            print("\n[DATA] Preprocessing data...\n")
            self.preprocess()
        else:
            print("\n[DATA] Loading preprocessed data...\n")
            # If the data is less than 1000, preprocess it again (this is because of the test mode)

            with h5py.File(self.preprocessed_data_path, 'r') as hf:
                total_length = len(hf)

            if total_length < 3000 and not self.test:
                print("\n[DATA] Preprocessing data again due to test mode...\n")
                self.preprocess()

        if self.test:
            print(f"\n[DATA] Limit data to maks \"50\" in \"Test Mode\"")
            if self.stage != "test":
                self.index_map = self.index_map[:50]

        print(f"\n[DATA] {self.stage} data: {len(self.index_map)}")
        print()


    def parse_raw_data(self):
        print()

        print("\n[DATA] Processed data not found or recreate = True, parsing raw data...\n")
        os.makedirs(os.path.dirname(self.parsed_data_path), exist_ok=True)

        print("\n[DATA] Parsing raw data...\n")
        seqxgpt_bench_parse_data(self.raw_data_path, self.parsed_data_path)

        print("\n[DATA] Done parsing data\n")
        print()


    def filter_data(self, stage, train_sources, test_sources):
        index_map = []

        with h5py.File(self.preprocessed_data_path, 'r') as hf:
            for key in tqdm(hf.keys(), desc="Scanning HDF5 attributes"):
                item_stage = hf[key].attrs['stage'].decode('utf-8')
                item_source = hf[key].attrs['source'].decode('utf-8')

                if stage != "test":
                    if item_stage == stage and item_source in train_sources:
                        index_map.append(key)

                if stage == "test":
                    if item_stage == stage and item_source in test_sources:
                        index_map.append(key)

        return index_map


    def generate_incoherent_doc(self, sentences, total):
        max_tries = 100

        incoherent_docs_sentence_index = []
        coherent_doc_sentence_index = list(range(len(sentences)))

        if len(sentences) < 2:
            return coherent_doc_sentence_index, None

        for i in range(total):
            tries = 0

            while tries < max_tries:
                incoherent_doc_sentence_index = coherent_doc_sentence_index.copy()
                random.shuffle(incoherent_doc_sentence_index)

                if incoherent_doc_sentence_index != coherent_doc_sentence_index:
                    if incoherent_doc_sentence_index not in incoherent_docs_sentence_index:
                        incoherent_docs_sentence_index.append(incoherent_doc_sentence_index)
                        break

                tries += 1

        if len(incoherent_docs_sentence_index) < total:
            while len(incoherent_docs_sentence_index) < total:
                random_index = random.randint(0, len(incoherent_docs_sentence_index) - 1)
                incoherent_docs_sentence_index.append(incoherent_docs_sentence_index[random_index])

        return coherent_doc_sentence_index, incoherent_docs_sentence_index


    def generate_incoherent_triplet(self, sentences, target_center_sentence_index, total):

        if len(sentences) < 3:
            return sentences, None

        max_tries = 100

        incoherent_triplets = []
        coherent_triplet = [target_center_sentence_index]

        left_index = target_center_sentence_index - 1
        right_index = target_center_sentence_index + 1

        if left_index >= 0:
            coherent_triplet.insert(0, left_index)

        if right_index < len(sentences):
            coherent_triplet.append(right_index)

        for i in range(total):
            tries = 0

            while tries < max_tries:
                incoherent_triplet = [target_center_sentence_index]

                random_left_index = random.randint(0, len(sentences) - 1)
                random_right_index = random.randint(0, len(sentences) - 1)

                if target_center_sentence_index != 0:
                    if random_left_index != target_center_sentence_index:
                        incoherent_triplet.insert(0, random_left_index)

                if target_center_sentence_index != len(sentences) - 1:
                    if random_right_index != target_center_sentence_index:
                        incoherent_triplet.append(random_right_index)

                # if random_left_index and random_right_index:
                #     print(incoherent_triplet, "<=>", coherent_triplet)

                if incoherent_triplet != coherent_triplet:
                    if incoherent_triplet not in incoherent_triplets:
                        if len(incoherent_triplet) == len(coherent_triplet):
                            incoherent_triplets.append(incoherent_triplet)
                            break

                tries += 1

        if len(incoherent_triplets) < total:
            for i in range(total - len(incoherent_triplets)):
                random_index = random.randint(0, len(incoherent_triplets) - 1)
                incoherent_triplets.append(incoherent_triplets[random_index])

        return coherent_triplet, incoherent_triplets


    def truncate_doc(self, sentence_index, sentences_token_length, target_sentence_index):
        doc = []
        total_token_length = 0

        aligned_target_sentence_index = sentence_index.index(target_sentence_index)

        doc.append(target_sentence_index)
        total_token_length += sentences_token_length[aligned_target_sentence_index]


        for i in range(len(sentence_index)):
            if i == 0:
                continue

            left_index = aligned_target_sentence_index - i
            right_index = aligned_target_sentence_index + i

            if left_index >= 0:

                if total_token_length + sentences_token_length[left_index] > self.max_len:
                    break

                doc.insert(0, sentence_index[left_index])
                total_token_length += sentences_token_length[left_index]

            if right_index < len(sentence_index):

                if total_token_length + sentences_token_length[right_index] > self.max_len:
                    break

                doc.append(sentence_index[right_index])
                total_token_length += sentences_token_length[right_index]

        return doc


    def build_doc(self, sentences, sentence_index, target_sentence_index):
        sentences_copy = sentences.copy()
        sentences_copy[target_sentence_index] = f"{self.tokenizer.sep_token}{sentences_copy[target_sentence_index]}{self.tokenizer.sep_token}"
        doc = " ".join([sentences_copy[i] for i in sentence_index])
        return doc


    def build_triplet(self, sentences, triplet, target_sentence_index):
        sentences_copy = sentences.copy()
        sentences_copy[target_sentence_index] = f"{self.tokenizer.sep_token}{sentences_copy[target_sentence_index]}{self.tokenizer.sep_token}"

        triplet_doc = []
        for i, _ in enumerate(triplet):
            triplet_doc.append(sentences_copy[i])
        triplet_doc = " ".join(triplet_doc)

        return triplet_doc


    def encode_input(self, input):
        encoded_input = self.tokenizer(input, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, padding_side="right", padding='max_length', truncation=True)

        input_ids = encoded_input['input_ids'][0]
        attention_mask = encoded_input['attention_mask'][0]
        token_type_ids = encoded_input['token_type_ids'][0]

        return torch.stack([input_ids, attention_mask, token_type_ids])


    def encode_inputs(self, inputs):
        if len(inputs) == 0:
            return torch.zeros(3, self.max_len)

        encoded_input = self.tokenizer(inputs, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, padding_side="right", padding='max_length', truncation=True)

        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        token_type_ids = encoded_input['token_type_ids']

        return torch.stack([input_ids, attention_mask, token_type_ids])


    def encode_cefr_representation(self, sentences, truncated_doc):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        sentence_idxs = []

        sep_token = self.tokenizer.sep_token

        for sentence_index in truncated_doc:
            sentence = sentences[sentence_index]
            encoded_input = self.tokenizer(sentence, add_special_tokens=False)

            input_ids.extend(encoded_input['input_ids'])
            attention_mask.extend(encoded_input['attention_mask'])
            token_type_ids.extend(encoded_input['token_type_ids'])
            sentence_idxs.extend([sentence_index] * len(encoded_input['input_ids']))

        max_len_wo_special_tokens = self.max_len - 3

        if len(input_ids) > max_len_wo_special_tokens:
            input_ids = input_ids[:max_len_wo_special_tokens]
            attention_mask = attention_mask[:max_len_wo_special_tokens]
            token_type_ids = token_type_ids[:max_len_wo_special_tokens]
            sentence_idxs = sentence_idxs[:max_len_wo_special_tokens]

        # add special tokens
        input_ids = [self.tokenizer.cls_token_id] + [self.local_lexical_special_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] + [1] + attention_mask + [1]
        token_type_ids = [0] + [0] + token_type_ids + [0]
        sentence_idxs = [-1] + [-1] + sentence_idxs + [-1]

        if len(input_ids) < self.max_len:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            attention_mask += [0] * (self.max_len - len(attention_mask))
            token_type_ids += [0] * (self.max_len - len(token_type_ids))
            sentence_idxs += [-1] * (self.max_len - len(sentence_idxs))

        return torch.stack([torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(sentence_idxs)])


    def truncate_global_cefr(self, word_cefr_levels, word_sentence_idx, focus_sentence_idx):

        if focus_sentence_idx not in word_sentence_idx:
            offet = 1
            while focus_sentence_idx not in word_sentence_idx:
                new_focus_sentence_idx = focus_sentence_idx + offet
                if new_focus_sentence_idx in word_sentence_idx:
                    focus_sentence_idx = new_focus_sentence_idx
                    break

                new_focus_sentence_idx = focus_sentence_idx - offet
                if new_focus_sentence_idx in word_sentence_idx:
                    focus_sentence_idx = new_focus_sentence_idx
                    break

                offet += 1


        focus_indices = [i for i, sentence_id in enumerate(word_sentence_idx)
                        if sentence_id == focus_sentence_idx]

        center_focus_idx = focus_indices[len(focus_indices) // 2]

        num_left = math.floor((self.max_len - 1) / 2)
        num_right = math.ceil((self.max_len - 1) / 2)

        ideal_start_idx = center_focus_idx - num_left
        ideal_end_idx = center_focus_idx + num_right

        list_len = len(word_sentence_idx)
        actual_start_idx = ideal_start_idx
        actual_end_idx = ideal_end_idx # Inclusive

        if actual_start_idx < 0:
            shift = 0 - actual_start_idx
            actual_start_idx = 0
            actual_end_idx += shift

        if actual_end_idx >= list_len:
            shift = actual_end_idx - (list_len - 1) # Amount we went past the end
            actual_end_idx = list_len - 1 # Clamp to the last valid index
            actual_start_idx -= shift # Shift the start index further left

        actual_start_idx = max(0, actual_start_idx)

        actual_end_slice = min(list_len, actual_end_idx + 1)

        if actual_end_slice - actual_start_idx < self.max_len:
            actual_start_idx = max(0, actual_end_slice - self.max_len)

        result_indices = list(range(actual_start_idx, actual_end_slice))

        final_word_cefr_levels = [value for i, value in enumerate(word_cefr_levels) if i in result_indices]

        if len(final_word_cefr_levels) > self.max_len:
            final_word_cefr_levels = final_word_cefr_levels[:self.max_len]
        elif len(final_word_cefr_levels) < self.max_len:
            final_word_cefr_levels += [-1] * (self.max_len - len(final_word_cefr_levels))

        return final_word_cefr_levels


    def truncate_local_cefr(self, sentence_cefr_levels, focus_sentence_idx):
        max_left = 63
        max_right = 64

        total_sentences = len(sentence_cefr_levels)
        optimal_left = focus_sentence_idx - max_left
        optimal_right = focus_sentence_idx + max_right

        if optimal_left < 0:
            shift = 0 - optimal_left
            optimal_left = 0
            optimal_right += shift

        if optimal_right >= total_sentences:
            shift = optimal_right - (total_sentences - 1)
            optimal_right = total_sentences - 1
            optimal_left -= shift

        actual_left = max(0, optimal_left)
        actual_right = min(total_sentences - 1, optimal_right)

        return sentence_cefr_levels[actual_left:actual_right + 1]


    def single_preprocess(self, data, cefr_analyzer):
        stage = data['train_ix']
        source = data['sources']
        labels = data['labels']

        sentences = data['sentences']

        sentences_token_length = [len(self.tokenizer.tokenize(sentence)) for sentence in sentences]

        coherencent_doc_sentence_index, incoherent_docs_sentence_index = self.generate_incoherent_doc(sentences, 3)

        ### Lexical ###
        sentence_cefr_levels = []
        word_cefr_levels = []
        word_sentence_idx = []

        for sentence_idx, sentence in enumerate(sentences):
            ascii_bytes = sentence.encode('ascii', 'ignore')
            cleaned_text = ascii_bytes.decode('utf-8')
            try:
                tokens = cefr_analyzer.analize_doc(nlp(cleaned_text))
            except Exception as e:
                print(traceback.format_exc())
                print(cleaned_text)
                sys.exit()

            cefr_data = []
            for token in tokens:
                _, _, is_skipped, level, _, _ = token
                if not is_skipped:
                    cefr_data.append(level)
                    word_cefr_levels.append(level)
                    word_sentence_idx.append(sentence_idx)

            sentence_cefr_level = (sum(cefr_data) / len(cefr_data)) if len(cefr_data) > 0 else 1
            sentence_cefr_levels.append(sentence_cefr_level)

        if len(sentence_cefr_levels) < 128:
            sentence_cefr_levels += [-1] * (128 - len(sentence_cefr_levels))

        global_cefr_level = (sum(word_cefr_levels) / len(word_cefr_levels)) if len(word_cefr_levels) > 0 else 1

        data_final = []

        #### Coherence ####
        for i in range(len(sentences)):

            # Global Coherence
            truncated_coherent_doc = self.truncate_doc(coherencent_doc_sentence_index, sentences_token_length, i)
            if incoherent_docs_sentence_index is not None:
                truncated_incoherent_docs = [self.truncate_doc(incoherent_docs_sentence_index[j], sentences_token_length, i) for j in range(len(incoherent_docs_sentence_index))]
            else:
                truncated_incoherent_docs = None

            coherent_doc = "[GLOBAL COHERENCE]" + self.build_doc(sentences, truncated_coherent_doc, i)
            if truncated_incoherent_docs is not None:
                incoherent_docs = ["[GLOBAL COHERENCE]" + self.build_doc(sentences, truncated_incoherent_docs[j], i) for j in range(len(truncated_incoherent_docs))]
            else:
                incoherent_docs = []

            # Local Coherence
            coherent_triplet, incoherent_triplets = self.generate_incoherent_triplet(sentences, i, 3)
            coherent_triplet = "[LOCAL COHERENCE]" + self.build_triplet(sentences, coherent_triplet, i)
            if incoherent_triplets is not None:
                incoherent_triplets = ["[LOCAL COHERENCE]" + self.build_triplet(sentences, incoherent_triplets[j], i) for j in range(len(incoherent_triplets))]
            else:
                incoherent_triplets = []


            # Global Lexical
            global_cefr_levels = self.truncate_global_cefr(word_cefr_levels, word_sentence_idx, i)

            # Local Lexical
            local_cefr_levels = [sentence_idx for sentence_idx, x in enumerate(sentence_cefr_levels) if round(x) == round(sentence_cefr_levels[i])]
            local_cefr_levels = self.truncate_local_cefr(local_cefr_levels, i)
            local_cefr_levels += [-1] * (128 - len(local_cefr_levels)) # padding

            label = torch.tensor(self.label_mapping[labels[i]])
            sentence = "[REPRESENTATION]" + sentences[i]

            #### Encode Inputs ####
            coherent_doc_input = self.encode_input(coherent_doc)
            incoherent_docs_input = self.encode_inputs(incoherent_docs)

            coherent_triplet_input = self.encode_input(coherent_triplet)
            incoherent_triplets_input = self.encode_inputs(incoherent_triplets)

            global_cefr_levels = torch.tensor(global_cefr_levels)
            local_cefr_levels = torch.tensor(local_cefr_levels)

            cefr_input = self.encode_cefr_representation(sentences, truncated_coherent_doc)

            sentence_input = self.encode_input(sentence)

            data_final.append({
                'stage': stage,
                'label': label,
                'source': source,
                'coherent_doc_input': coherent_doc_input,
                'incoherent_docs_input': incoherent_docs_input,
                'coherent_triplet_input': coherent_triplet_input,
                'incoherent_triplets_input': incoherent_triplets_input,
                'global_cefr_levels': global_cefr_levels,
                'global_cefr_level': global_cefr_level,
                'local_cefr_levels': local_cefr_levels,
                'local_cefr_level': sentence_cefr_levels[i],
                'cefr_input': cefr_input,
                'sentence_input': sentence_input,
            })

        return data_final


    def preprocess(self):
        raw_data = self.raw_data

        if self.test:
            train_data = []
            valid_data = []
            test_data = []
            for data in raw_data:
                if len(train_data) < 50 and data["train_ix"] == "train":
                    train_data.append(data)
                elif len(valid_data) < 50 and data["train_ix"] == "valid":
                    valid_data.append(data)
                elif len(test_data) < 50 and data["train_ix"] == "test":
                    test_data.append(data)
                if len(train_data) == 50 and len(valid_data) == 50 and len(test_data) == 50:
                    break
            raw_data = train_data + valid_data + test_data

        preprocessed_data = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_preprocess_worker)(data, self.abbreviation_mapping, self.tokenizer_config)
            for data in tqdm(raw_data, desc="Preprocessing data")
        )

        flat_data = [item for sublist in preprocessed_data for item in sublist]

        os.makedirs(os.path.dirname(self.preprocessed_data_path), exist_ok=True)

        with h5py.File(self.preprocessed_data_path, 'w') as hf:
            for i, item_dict in enumerate(tqdm(flat_data, desc="Saving to HDF5")):
                grp = hf.create_group(str(i))

                for key, value in item_dict.items():
                    if isinstance(value, str):
                        grp.attrs[key] = np.bytes_(value)
                    elif isinstance(value, torch.Tensor):
                        if value.ndim > 0:
                            grp.create_dataset(key, data=value.numpy(), compression="gzip")
                        else:
                            grp.create_dataset(key, data=value.numpy())
                    elif isinstance(value, (float, int)):
                        grp.create_dataset(key, data=value)



    def setup(self, stage=None):

        if stage=='fit':
            self.h5_file_train = h5py.File(self.preprocessed_data_path, 'r')
            self.h5_file_val = h5py.File(self.preprocessed_data_path, 'r')
            self.train_filtered = self.filter_data("train", self.train_sources, self.test_sources)
            self.val_filtered = self.filter_data("valid", self.train_sources, self.test_sources)
            self.train_dataset = GLCliCSeqXGPTBenchDataset(self.train_filtered, self.source_mapping, "train", self.h5_file_train, self.model_config)
            self.val_dataset = GLCliCSeqXGPTBenchDataset(self.val_filtered, self.source_mapping, "valid", self.h5_file_val, self.model_config)
        elif stage=='test':
            self.h5_file_test = h5py.File(self.preprocessed_data_path, 'r')
            self.test_filtered = self.filter_data("test", self.train_sources, self.test_sources)
            self.test_dataset = GLCliCSeqXGPTBenchDataset(self.test_filtered, self.source_mapping, "test", self.h5_file_test, self.model_config)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count() if not self.test else 1)



class RobustCEFRSpaCyAnalyzer(CEFRSpaCyAnalyzer):

    # Override the original method to handle unknown words
    def _fetch_word_pos_level_tokens(self, word_pos_tokens_set: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
        result_dict = dict()
        for word, pos_tag in word_pos_tokens_set:
            try:
                level = self._analyzer.get_word_pos_level_float(word, pos_tag, avg_level_not_found_pos=True)
            except IndexError:
                level = 0
            result_dict[(word, pos_tag)] = level if level is not None else 0

        return result_dict
