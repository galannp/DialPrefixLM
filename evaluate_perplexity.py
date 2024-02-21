# Setup
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

# Imports
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from DialPrefixLM.model import build_model_tokenizer
from DialPrefixLM.preprocess.build_dialogue_datasets import build_single_dataset
from DialPrefixLM.utils import print_gpu_utilization

def evaluate_perplexity(MODEL_CHECKPOINT, DATASET, NB_TEST_EXAMPLES):
    # Constants
    DS_PARAMS = [
            {"path": "ccdv/mediasum", "split":"validation", "extract_cols":["document"]},
            {"path": "TanveerAman/AMI-Corpus-Text-Summarization", "split": "train", "extract_cols": ["Dialogue"]},
            {"path": "knkarthick/dialogsum", "split": "validation", "extract_cols": ["dialogue"]},
            #{"path": "Salesforce/dialogstudio", "name": "SODA", "split": "test", "extract_cols": extract_dialogues_soda}
    ]
    TEST_DS = [ds_params for ds_params in DS_PARAMS if ds_params["path"] == DATASET][0]
    RENAMED_COLS = ["dialogue"]

    DEVICE = "cuda"
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 2048


    # Load tested model and tokenizer
    model, tokenizer = build_model_tokenizer(MODEL_CHECKPOINT, device_map="cuda:0")


    def preprocess_perplexity(examples, context_length):
        tk_examples = tokenizer(examples["dialogue"], truncation=True,
                        max_length=context_length, return_overflowing_tokens=True)
        del tk_examples["overflow_to_sample_mapping"]
        return tk_examples

    # Load test Dataset
    test_ds = build_single_dataset(**TEST_DS, renamed_cols=RENAMED_COLS)
    if NB_TEST_EXAMPLES is not None:
        test_ds = test_ds.shuffle().select(range(min(NB_TEST_EXAMPLES, len(test_ds))))
    print(test_ds)

    # Preprocess test Dataset
    tokenized_test_ds = test_ds.map(lambda examples: preprocess_perplexity(examples, CONTEXT_LENGTH),
                                                batched=True, remove_columns=RENAMED_COLS, num_proc=4)
    print(len(test_ds), len(tokenized_test_ds))

    """from DialPrefixLM.preprocess.inject_noise_to_dialogue import inject_dialogue_noise

    def preprocess(examples, context_length, inject_dial_noise=False):
        # TODO: change name of variables and keep consistancy within the notebook (bopref) doesn't mean anything anymore
        # TODO: add eopref and bopref when one span is split between two samples from the batch
        if inject_dial_noise:
            # To fit to the chosen LLM
            spans, idx_noised = inject_dialogue_noise(examples["dialogue"], bos_token=tokenizer.bos_token)
            spans_ids = tokenizer(spans, truncation=False, add_special_tokens=False)["input_ids"]
            spans_mask = [[span_type] * len(span_tk) for span_tk, span_type in zip(spans_ids, idx_noised)]
            attention_mask = [[1] * len(span_tk) for span_tk in spans_ids]

            tk_examples = {
                "input_ids": spans_ids,
                "attention_mask": attention_mask,
                "prefix_mask": spans_mask
            }
        else:
            tk_examples = tokenizer(examples["dialogue"], truncation=False)

        def flatten_list(l):
            res = []
            for elt in l:
                res += elt
            return res

        tk_examples = {k: flatten_list(tk_examples[k]) for k in tk_examples.keys()}
        total_length = len(tk_examples["input_ids"])
        tk_examples = {
                k: [t[i : i + context_length] for i in range(0, total_length, context_length)]
                for k, t in tk_examples.items()
        }

        return tk_examples

    # Load test Dataset
    test_ds = build_single_dataset(**TEST_DS, renamed_cols=RENAMED_COLS)
    test_ds = test_ds.shuffle()
    test_ds = test_ds.select(range(min(NB_TEST_EXAMPLES, len(test_ds))))
    print(test_ds)

    # Preprocess test Dataset
    preproc = lambda examples: preprocess(examples, CONTEXT_LENGTH, inject_dial_noise=True)
    tokenized_test_ds = test_ds.map(preproc, batched=True, remove_columns=RENAMED_COLS, num_proc=4)
    print(len(test_ds), len(tokenized_test_ds))"""


    # Prepare test Dataset
    test_data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    """from DialPrefixLM.preprocess.data_collators import DataCollatorForPrefixLM
    test_data_collator = DataCollatorForPrefixLM(tokenizer)"""

    test_dataloader = DataLoader(
        tokenized_test_ds,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=test_data_collator,
        pin_memory=True,
        num_workers=4
    )


    # Perplexity evaluation over Test dataset
    nlls = []
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            batch.to(DEVICE)
            outputs = model(**batch)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
    avg_logit = torch.stack(nlls).mean()
    ppl = torch.exp(avg_logit)
    print_gpu_utilization()

    print(f"Perplexity for dataset {TEST_DS['path']}: {ppl}")
    print(f"average logit for dataset {TEST_DS['path']}: {avg_logit}")


    # Theoritical experiment:
    # show that DialPrefixLM allows better generalization both to other dialogue domains and longer dialogue modelling!


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("--nb_test_examples", type=int)

    args = parser.parse_args()
    evaluate_perplexity(args.model_checkpoint, args.dataset, args.nb_test_examples)