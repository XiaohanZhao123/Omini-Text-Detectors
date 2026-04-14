import os
import sys
import math
import h5py
import spacy
import torch
import pickle
import random
import joblib
import traceback
import numpy as np

from utils.config import GLCLiCModelConfig
from transformers import AutoTokenizer
from cefrpy import CEFRSpaCyAnalyzer
from torch.utils.data import Dataset
from tqdm import tqdm



nlp = spacy.load("en_core_web_sm")


class GLCliCCoAuthorDataset(Dataset):
    def __init__(self, data, model_config: GLCLiCModelConfig):
        random.seed(model_config.seed)

        self.data = data
        self.model_config = model_config


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        label = self.data[idx]['label']
        sentence_input = self.data[idx]['sentence_input']
        essay_type = self.type_mapping[self.data[idx]['type']]

        all_inputs = {}

        if self.model_config.global_coherence:
            coherent_doc_input = self.data[idx]['coherent_doc_input']
            incoherent_docs_input = self.data[idx]['incoherent_docs_input']

        if self.model_config.local_coherence:
            coherent_triplet_input = self.data[idx]['coherent_triplet_input']
            incoherent_triplets_input = self.data[idx]['incoherent_triplets_input']

        if self.model_config.global_lexical:
            global_cefr_levels = self.data[idx]['global_cefr_levels']
            global_cefr_level = self.data[idx]['global_cefr_level']

        if self.model_config.local_lexical:
            local_cefr_levels = self.data[idx]['local_cefr_levels']
            cefr_input = self.data[idx]['cefr_input']
            local_cefr_level = self.data[idx]['local_cefr_level']

        all_inputs["labels"] = label.float()
        all_inputs["sentence_inputs"] = sentence_input
        all_inputs["types"] = essay_type

        if self.model_config.global_coherence:
            all_inputs["global_coherences"] = (coherent_doc_input, incoherent_docs_input)

        if self.model_config.local_coherence:
            all_inputs["local_coherences"] = (coherent_triplet_input, incoherent_triplets_input)

        if self.model_config.global_lexical:
            all_inputs["global_lexical"] = global_cefr_levels
            all_inputs["global_cefr_level"] = global_cefr_level

        if self.model_config.local_lexical:
            all_inputs["local_lexical"] = (local_cefr_levels, cefr_input)
            all_inputs["local_cefr_level"] = local_cefr_level

        return all_inputs


class GLCliCSeqXGPTBenchDataset(Dataset):
    def __init__(self, index_map, source_mapping, stage, h5_file, model_config: GLCLiCModelConfig):
        random.seed(model_config.seed)
        self.index_map = index_map
        self.source_mapping = source_mapping
        self.stage = stage
        self.h5_file = h5_file
        self.model_config = model_config
        self.device = torch.device("cuda" if self.model_config.use_gpu else "cpu")


    def __len__(self):
        return len(self.index_map)


    def __getitem__(self, idx):
        group_key = self.index_map[idx]
        data_h5 = self.h5_file[group_key]

        data = {}
        for key, dataset in data_h5.items():
            if dataset.ndim == 0:
                value = dataset[()]
            else:
                value = dataset[:]

            if key in ["global_cefr_levels", "global_cefr_level", "local_cefr_level"]:
                data[key] = torch.tensor(value, dtype=torch.float)
            else:
                data[key] = torch.tensor(value, dtype=torch.long)

        for key, value in data_h5.attrs.items():
            data[key] = value.decode('utf-8')

        label = data['label']
        sentence_input = data['sentence_input']
        source = self.source_mapping[data['source']]

        all_inputs = {}

        if self.model_config.global_coherence:
            coherent_doc_input = data['coherent_doc_input']
            incoherent_docs_input = data['incoherent_docs_input']

        if self.model_config.local_coherence:
            coherent_triplet_input = data['coherent_triplet_input']
            incoherent_triplets_input = data['incoherent_triplets_input']

        if self.model_config.global_lexical:
            global_cefr_levels = data['global_cefr_levels']
            global_cefr_level = data['global_cefr_level']

        if self.model_config.local_lexical:
            local_cefr_levels = data['local_cefr_levels']
            cefr_input = data['cefr_input']
            local_cefr_level = data['local_cefr_level']

        all_inputs["labels"] = label.float().squeeze()
        all_inputs["sentence_inputs"] = sentence_input
        all_inputs["sources"] = source

        if self.stage != "test":
            if self.model_config.global_coherence:
                all_inputs["global_coherences"] = (coherent_doc_input, incoherent_docs_input)

            if self.model_config.local_coherence:
                all_inputs["local_coherences"] = (coherent_triplet_input, incoherent_triplets_input)

            if self.model_config.global_lexical:
                all_inputs["global_lexical"] = global_cefr_levels
                all_inputs["global_cefr_level"] = global_cefr_level

            if self.model_config.local_lexical:
                all_inputs["local_lexical"] = (local_cefr_levels, cefr_input)
                all_inputs["local_cefr_level"] = local_cefr_level
        else:
            if self.model_config.global_coherence:
                all_inputs["global_coherences"] = (coherent_doc_input, [])

            if self.model_config.local_coherence:
                all_inputs["local_coherences"] = (coherent_triplet_input, [])

            if self.model_config.global_lexical:
                all_inputs["global_lexical"] = global_cefr_levels
                all_inputs["global_cefr_level"] = global_cefr_level

            if self.model_config.local_lexical:
                all_inputs["local_lexical"] = (local_cefr_levels, cefr_input)
                all_inputs["local_cefr_level"] = local_cefr_level

        return all_inputs
