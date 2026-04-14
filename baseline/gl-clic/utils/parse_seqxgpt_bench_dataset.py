import os
import sys
import nltk
import json
import random
import pickle
import joblib
import pandas as pd

from tqdm import tqdm
from collections import Counter


def process_single_session(session_data, stage):
    sentences = []
    labels = []

    sent_label, sent_text = get_sentence_with_label(session_data)

    for sent_label, sent_text in zip(sent_label, sent_text):
        sentences.append(sent_text)
        labels.append(sent_label)

    if stage != "test" and len(sentences) < 3:
        return None

    return {
        "sentences": sentences,
        "labels": labels,
        "sources": session_data["label"] if session_data["label"] != "gpt3re" else "gpt3",
        "train_ix": stage
    }


# Modified from https://github.com/Jihuai-wpy/SeqXGPT/blob/main/SeqXGPT/SeqXGPT/train.py
def split_dataset(data_path, train_ratio=0.9):
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    total_samples = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)

    import random
    random.seed(0)
    random.shuffle(total_samples)

    split_index = int(len(total_samples) * train_ratio)
    train_data = total_samples[:split_index]
    test_data = total_samples[split_index:]

    val_split_index = int(len(train_data) * 0.1)
    val_data = train_data[:val_split_index]
    train_data = train_data[val_split_index:]

    return train_data, val_data, test_data


# Modified from https://github.com/Jihuai-wpy/SeqXGPT/blob/main/SeqXGPT/SeqXGPT/train.py
def get_sentence_with_label(data):
    text = data["text"]

    if "prompt_len" in data:
        prompt_len = data["prompt_len"]
        pua_sequence = ["U"] * prompt_len
        pua_sequence += ["A"] * (len(text) - prompt_len)
        pua_sequence = "".join(pua_sequence)
    else:
        pua_sequence = ["U"] * len(text)
        pua_sequence = "".join(pua_sequence)

    sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_separator.tokenize(text)

    offset = 0
    sent_label = []
    sent_text = []

    for sent in sents:
        start = text[offset:].find(sent) + offset
        end = start + len(sent)
        offset = end

        label_search = pua_sequence[start:end]
        most_common_tag = Counter(label_search).most_common(1)[0][0]

        sent_label.append(most_common_tag)
        sent_text.append(sent)

    return sent_label, sent_text


def parse_data(dir_path, save_path, num_workers=-2):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_data, val_data, test_data = split_dataset(dir_path)

    print(f"\nParsing train data [{len(train_data)}]")
    train_parsed_dataset = [joblib.Parallel(n_jobs=num_workers, verbose=1)(
        joblib.delayed(process_single_session)(data, "train") for data in train_data
    )]

    print(f"\nParsing val data [{len(val_data)}]")
    val_parsed_dataset = [joblib.Parallel(n_jobs=num_workers, verbose=1)(
        joblib.delayed(process_single_session)(data, "valid") for data in val_data
    )]

    print(f"\nParsing test data [{len(test_data)}]")
    test_parsed_dataset = [joblib.Parallel(n_jobs=num_workers, verbose=1)(
        joblib.delayed(process_single_session)(data, "test") for data in test_data
    )]

    train_parsed_dataset = [item for sublist in train_parsed_dataset for item in sublist]
    val_parsed_dataset = [item for sublist in val_parsed_dataset for item in sublist]
    test_parsed_dataset = [item for sublist in test_parsed_dataset for item in sublist]

    parsed_dataset = train_parsed_dataset + val_parsed_dataset + test_parsed_dataset

    # Remove None
    parsed_dataset = [item for item in parsed_dataset if item is not None]

    # Checking
    # less_than_3_sentences_count = {}
    # less_than_3_sentences_source = {}
    # less_than_3_sentences_train_ix = {}

    # for data in parsed_dataset:
    #     if len(data["sentences"]) < 3:
    #         if len(data["sentences"]) not in less_than_3_sentences_count:
    #             less_than_3_sentences_count[len(data["sentences"])] = 0

    #         if data["sources"] not in less_than_3_sentences_source:
    #             less_than_3_sentences_source[data["sources"]] = 0

    #         if data["train_ix"] not in less_than_3_sentences_train_ix:
    #             less_than_3_sentences_train_ix[data["train_ix"]] = 0

    #         less_than_3_sentences_count[len(data["sentences"])] += 1
    #         less_than_3_sentences_source[data["sources"]] += 1
    #         less_than_3_sentences_train_ix[data["train_ix"]] += 1

    pickle.dump(parsed_dataset, open(save_path, 'wb'))


# if __name__ == "__main__":
#     parse_data("dataset/CoAuthor/raw/coauthor.csv", "dataset/CoAuthor/parsed/coauthor.pkl")
