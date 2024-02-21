import json
from datasets import Dataset, load_dataset


def build_single_dataset(path, filter_short_dials=True, extract_cols=None, build_cols=None, renamed_cols=None, **kwargs):
    ds = load_dataset(path, **kwargs)
    if extract_cols is not None:
        ds = ds.select_columns(extract_cols)
    if build_cols is not None:
        ds = ds.map(build_cols, num_proc=4)
    if renamed_cols is not None:
        ds = ds.rename_columns(dict(zip(ds.column_names, renamed_cols)))

    ds = ds.map(preproc_dialogues, batched=True, num_proc=4)

    if filter_short_dials:
        nb_dialogues_original = len(ds)
        print(f"Nb dialogues originally in the dataset: {nb_dialogues_original}")
        ds = ds.filter(short_dialogues, batched=True, num_proc=4)
        print(f"Nb dialogues filtered out because too short: {nb_dialogues_original - len(ds)}")
    return ds


def preproc_dialogues(examples):
    examples["dialogue"] = [dial.replace("</s>", "\n") for dial in examples["dialogue"]]
    return examples


def short_dialogues(examples, threshold_len_dial=400):
    return [len(dial) >= threshold_len_dial for dial in examples["dialogue"]]


def extract_dialogues_soda(ds, name_cols):
    assert len(name_cols) == 1
    output_col = name_cols[0]
    dialogues = []

    nb_wrong = 0
    for i, ex in enumerate(ds):
        speakers = json.loads(ex["original dialog info"])["speakers"]
        speaker_and_turn_0 = ex["log"][0]["user utterance"].split(":")
        if speaker_and_turn_0[-1].strip() == "":
            nb_wrong += 1
            continue
        utterances = []
        if len(speaker_and_turn_0) > 1:
            speaker_user = speaker_and_turn_0[-2].strip()
            for s in speakers:
                if s != speaker_user:
                    speaker_system = s
        else:
            speaker_user = speakers[0]
            speaker_system = speakers[1]

        nb_log = len(ex["log"])
        is_wrong = False
        for j, log in enumerate(ex["log"]):
            user_utterance = log["user utterance"]
            system_response = log["system response"]
            if j > 0 and ":" in user_utterance or ":" in system_response or j < nb_log - 1 and system_response == "" or user_utterance == "":
                is_wrong = True
                nb_wrong += 1
                break
            if ":" in user_utterance:
                user_utterance = user_utterance.split(":")[-1]
            user_utterance = speaker_user + ": " + user_utterance
            if system_response != "":
                system_response = speaker_system + ": " + system_response
                utterances.extend([user_utterance, system_response])
            else:
                utterances.append(user_utterance)
        if not is_wrong:
            dialogues.append("\n".join(utterances))
    print(nb_wrong / len(ds))

    return Dataset.from_dict({output_col: dialogues})


def extract_features_dream(example):
    example["dialogue"] = "\n".join(example["dialogue"])
    example["label_ans"] = example["choice"].index(example["answer"])
    return example


def extract_features_mutual(example):
    example["label_ans"] = ord(example["answers"]) - ord("A")
    return example