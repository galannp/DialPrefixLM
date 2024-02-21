# Setup
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

# Imports
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator
from DialPrefixLM.model import build_model_tokenizer
from DialPrefixLM.preprocess.build_dialogue_datasets import build_single_dataset, extract_features_dream, extract_features_mutual
from DialPrefixLM.utils import print_gpu_utilization
from DialPrefixLM.metrics import compute_rouge_eval
import torch
import re

def evaluate_qa(MODEL_CHECKPOINT, DATASET, NB_TEST_EXAMPLES):
    # Constants
    DS_PARAMS = [
        {"path": "dream", "split": "train", "extract_cols": ["dialogue", "question", "choice", "answer"], "build_cols": extract_features_dream},
        {"path": "EleutherAI/mutual", "name": "mutual_plus", "split": "train", "extract_cols": ["article", "options", "answers"], "build_cols": extract_features_mutual}
    ]
    TEST_DS = [ds_params for ds_params in DS_PARAMS if ds_params["path"] == DATASET][0]

    #RENAMED_COLS = ["dialogue", "question", "choice", "answer", "label_ans"]
    RENAMED_COLS = ["dialogue", "choice", "answer", "label_ans"]

    DEVICE = "cuda"
    BATCH_SIZE = 1 #8
    MAX_TARGET_LENGTH = 128
    STEP_BY_STEP = False


    # Load tested model and tokenizer
    model, tokenizer = build_model_tokenizer(MODEL_CHECKPOINT, device_map="cuda:0")

    def get_labels(nb_choice):
        return [chr(ord("A") + i) for i in range(nb_choice)]

    def preprocess_qa(example):
        if "question" in example:
            question_str = "Question: " + example["question"]
        else:
            question_str = "Question: What is the next dialogue utterance?"
        choice_str = " ".join([ch_idx + ")" + " " + ch for ch_idx, ch in zip(get_labels(len(example["choice"])), example["choice"])])

        prompt_str = example["dialogue"] + "\n" + question_str + "\n" + choice_str + "\n" + "Answer:"
        if STEP_BY_STEP:
            prompt_str += " Let's think step by step."
        return {
            "prompt": prompt_str,
        }

    def tokenize(examples):
        return tokenizer(examples["prompt"], truncation=False)

    # Load test Dataset
    test_ds = build_single_dataset(**TEST_DS, filter_short_dials=False, renamed_cols=RENAMED_COLS)
    if NB_TEST_EXAMPLES is not None:
        test_ds = test_ds.shuffle().select(range(min(NB_TEST_EXAMPLES, len(test_ds))))

    # Preprocess test Dataset
    remove_columns = test_ds.column_names.copy()
    remove_columns.remove("label_ans")
    test_ds_preproc = test_ds.map(preprocess_qa, remove_columns=remove_columns, num_proc=4)
    tokenized_test_ds = test_ds_preproc.map(lambda examples: tokenize(examples), batched=True, remove_columns=["prompt"], num_proc=4)
    answer_choice = test_ds["choice"]


    # Prepare Evaluation
    test_dataloader = DataLoader(
        tokenized_test_ds,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=4
    )
    answer_choice_batched = [answer_choice[i: i + BATCH_SIZE] for i in range(0, len(answer_choice), BATCH_SIZE)]


    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnTokens(StoppingCriteria):
        def __init__(self, stops = []):
            StoppingCriteria.__init__(self)
            self.stops = stops

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            for stop_id in self.stops:
                if stop_id in tokenizer.decode(input_ids[0][-1]):
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stops = ["\n"])])


    # Evaluation over Test dataset

    matches = []
    model.eval()
    for batch, choice in tqdm(zip(test_dataloader, answer_choice_batched)):
        with torch.no_grad():
            prompt = batch["input_ids"]
            suppress_tokens = [tokenizer.eos_token_id, tokenizer(" ")["input_ids"][-1]]

            prompt_and_pred = model.generate(prompt.to(DEVICE), max_new_tokens=MAX_TARGET_LENGTH,
                                            begin_suppress_tokens=suppress_tokens, stopping_criteria=stopping_criteria)
            if STEP_BY_STEP:
                prompt = torch.tensor([prompt_and_pred[0].tolist() + tokenizer("\n" + "Therefore, the answer is:", add_special_tokens=False)["input_ids"]])
                prompt_and_pred = model.generate(prompt.to(DEVICE), max_new_tokens=MAX_TARGET_LENGTH,
                                                begin_suppress_tokens=suppress_tokens, stopping_criteria=stopping_criteria)
            print(tokenizer.decode(prompt_and_pred[0]).replace("\n", "\n\n"))
            predictions = tokenizer.decode(prompt_and_pred[:, prompt.shape[1]:][0]).strip()
            print(predictions)
            print(f"Groundtruth Answer: {batch['label_ans'][0]}\n")
            ans_pred_split = predictions.split(maxsplit=1)
            if len(ans_pred_split) > 0:
                ans_pred_str = ans_pred_split[0]
                choice_lower = [c.lower() for c in choice[0]]
                predictions_lower = predictions.lower()
                labels = get_labels(len(choice[0]))
                if len(predictions) == 1 and predictions in labels:
                    label_pred = labels.index(predictions)
                elif len(ans_pred_str) == 2 and ans_pred_str[0] in labels and not ans_pred_str[1].isalpha():
                    label_pred = labels.index(ans_pred_str[0])
                elif predictions_lower in choice_lower:
                    label_pred = choice_lower.index(predictions_lower)
                elif len(predictions) > 0 and predictions_lower[-1] == "." and predictions_lower[:-1] in choice_lower:
                    label_pred = choice_lower.index(predictions_lower[:-1])
                elif len(predictions) > 1 and predictions_lower[0] == predictions_lower[-1] == '"' and predictions_lower[1:-1] in choice_lower:
                    label_pred = choice_lower.index(predictions_lower[1:-1])
                elif len(predictions) > 2 and predictions_lower[0] == predictions_lower[-1] == '"' and predictions_lower[-2] == "." and  predictions_lower[1:-2] in choice_lower:
                    label_pred = choice_lower.index(predictions_lower[1:-2])
                elif len(ans_pred_split) > 1 and ans_pred_split[1].lower() in choice_lower:
                    label_pred = choice_lower.index(ans_pred_split[1].lower())
                else:
                    #TODO
                    print(111111111111111111)
                    continue
            else:
                #TODO
                print(22222222222222222222222222222)
                continue
            print(label_pred)
            matches.append(label_pred == batch["label_ans"])

    print_gpu_utilization()
    print(f"Accuracy on dataset {TEST_DS['path']}:\n\n{sum(matches) / len(matches)}")


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("--nb_test_examples", type=int)

    args = parser.parse_args()
    evaluate_qa(args.model_checkpoint, args.dataset, args.nb_test_examples)


# Split between short and long dialogue and see the difference of performance across models
# how to handle cases when the answer is not recognized by the parser
