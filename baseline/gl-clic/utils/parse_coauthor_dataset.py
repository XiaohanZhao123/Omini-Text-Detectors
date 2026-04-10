import os
import sys
import random
import pickle
import joblib
import pandas as pd

from tqdm import tqdm


def remove_prompt_char(char_type_seq, text):
  new_text = ""
  new_pua_sequence = ""
  for char, char_type in zip(text, char_type_seq):
    if char_type != 'P':
      new_text += char
      new_pua_sequence += char_type
  return new_text, new_pua_sequence


def process_single_session(session_data, session_id):
    session_data = session_data.sort_values(by=['sent_id'], ascending=True)

    sentences = []
    sentences_topic_id = []
    labels = []
    pua_sequences = []
    type = []
    train_ix = []

    for i in range(len(session_data)):
        # skip prompt sentences
        if session_data.iloc[i]['label'] == -1:
            continue

        sequence = session_data.iloc[i]['pua_sequence']
        sentence = session_data.iloc[i]['sentence_text']

        if len(sentence) != len(sequence):
            if len(sentence) > len(sequence):
                left_over = len(sentence) - len(sequence)
                last_sequence = sequence[-1]
                sequence += last_sequence * left_over
            else:
                sequence = sequence[:len(sentence)]

        # remove P type character from sentence text
        if "P" in sequence:
            sentence, sequence = remove_prompt_char(sequence, sentence)

        if len(sentence) != len(sequence):
            raise ValueError(f"Different length between sentence and sequence for session {session_id} and sentence {session_data.iloc[i]['sent_id']}")

        sentences.append(sentence)
        sentences_topic_id.append(session_data.iloc[i]['topic_label'])
        labels.append(session_data.iloc[i]['label'])
        pua_sequences.append(sequence)
        type.append(session_data.iloc[i]['essay_type'])
        train_ix.append(session_data.iloc[i]['train_ix'])

    if len(set(type)) != 1:
        raise ValueError(f"Types are not the same for session {session_id}")
    else:
        type = type[0]

    if len(set(train_ix)) != 1:
        raise ValueError(f"Train_ix are not the same for session {session_id}")
    else:
        train_ix = train_ix[0]

    return [{
        "session_id": session_id,
        "sentences": sentences,
        "sentences_topic_id": sentences_topic_id,
        "labels": labels,
        "pua_sequences": pua_sequences,
        "type": type,
        "train_ix": train_ix
    }]


def parse_data(file_path, save_path, num_workers=-2):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # data = pd.read_excel(file_path, sheet_name="Sheet1")
    data = pd.read_csv(file_path)
    data = data.sort_values(by=['session_id', 'sent_id'], ascending=True)

    session_ids = data['session_id'].unique()

    parsed_dataset = joblib.Parallel(n_jobs=num_workers, verbose=1)(
        joblib.delayed(process_single_session)(data[data['session_id'] == session_id], session_id) for session_id in session_ids
    )

    parsed_dataset = [item for sublist in parsed_dataset for item in sublist]

    pickle.dump(parsed_dataset, open(save_path, 'wb'))


if __name__ == "__main__":
    parse_data("dataset/CoAuthor/raw/coauthor.csv", "dataset/CoAuthor/parsed/coauthor.pkl")
