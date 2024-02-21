# Setup
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

# Imports
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import default_data_collator
from DialPrefixLM.model import build_model_tokenizer
from DialPrefixLM.preprocess.build_dialogue_datasets import build_single_dataset
from DialPrefixLM.utils import print_gpu_utilization
from DialPrefixLM.metrics import compute_rouge_eval
import torch

def evaluate_summarization(MODEL_CHECKPOINT, NB_TEST_EXAMPLES):
    # Constants
    #TEST_DS = {"path": "TanveerAman/AMI-Corpus-Text-Summarization", "split": "validation", "extract_cols": ["Dialogue", "Summaries"]}
    TEST_DS = {"path": "knkarthick/dialogsum", "split": "validation", "extract_cols": ["dialogue", "summary"]}
    #TEST_DS = {"path": "ccdv/mediasum", "split": "validation", "extract_cols": ["document", "summary"]}
    RENAMED_COLS = ["dialogue", "summary"]

    PROMPT = "\nSummarize:"

    DEVICE = "cuda"
    BATCH_SIZE = 1
    CONTEXT_LENGTH = 2048
    MAX_TARGET_LENGTH = 64 #32
    NB_FEW_SHOT = 0


    # Load tested model and tokenizer
    model, tokenizer = build_model_tokenizer(MODEL_CHECKPOINT, device_map="cuda:0")


    # Load test Dataset
    test_ds = build_single_dataset(**TEST_DS, renamed_cols=RENAMED_COLS)
    if NB_TEST_EXAMPLES is not None:
        test_ds = test_ds.select(range(NB_TEST_EXAMPLES))
    print(test_ds)


    def build_few_shot_context(few_shot_examples, prompt, tokenizer):
        assert len(few_shot_examples) > 1
        few_shot_context = ""
        for ex in few_shot_examples:
            few_shot_context += "Dialogue:\n" + ex["dialogue"] + prompt + ex["summary"] + tokenizer.eos_token + "\n\n"
        few_shot_context += "Dialogue:\n"
        return few_shot_context

    def preprocess_for_summ_test(examples, context_length, max_target_length, prompt="", few_shot_context=""):
        # Preprocess input
        inputs = [few_shot_context + doc + prompt for doc in examples["dialogue"]]
        if context_length is None:
            max_input_length = None
        else:
            max_input_length = context_length - max_target_length
        tk_examples = tokenizer(inputs, truncation=True, max_length=max_input_length)
        [print(111111111111111) for input in tk_examples["input_ids"] if len(input) == context_length]

        # Preprocess labels
        tk_examples["labels"] = tokenizer(examples["summary"], truncation=False)["input_ids"]

        return tk_examples



    few_shot_context = ""
    if NB_FEW_SHOT > 0:
        test_ds = test_ds.train_test_split(train_size=NB_FEW_SHOT, shuffle=False)
        few_shot_examples, test_ds = test_ds["train"], test_ds["test"]

        """# Only for Dialogsum
        few_shot_examples = list(few_shot_examples)
        few_shot_examples[0]["dialogue"] = few_shot_examples[0]["dialogue"].replace("#Person1#", "Dr. Anderson").replace("#Person2#", "Leo")
        few_shot_examples[0]["summary"] = few_shot_examples[0]["summary"].replace("#Person1#", "Dr. Anderson").replace("#Person2#", "Leo")
        if NB_FEW_SHOT > 1:
            few_shot_examples[1]["dialogue"] = few_shot_examples[1]["dialogue"].replace("#Person1#", "Lucy").replace("#Person2#", "Jimmy")
            few_shot_examples[1]["summary"] = few_shot_examples[1]["summary"].replace("#Person1#", "Lucy").replace("#Person2#", "Jimmy")
        if NB_FEW_SHOT > 2:
            few_shot_examples[2]["dialogue"] = few_shot_examples[2]["dialogue"].replace("#Person1#", "Thomas").replace("#Person2#", "Jules")
            few_shot_examples[2]["summary"] = few_shot_examples[2]["summary"].replace("#Person1#", "Thomas").replace("#Person2#", "Jules")"""

        few_shot_context = build_few_shot_context(few_shot_examples, prompt=PROMPT, tokenizer=tokenizer)
        print(few_shot_context)



    # Preprocess test Dataset
    tokenized_test_ds = test_ds.map(lambda examples: preprocess_for_summ_test(examples, CONTEXT_LENGTH, MAX_TARGET_LENGTH, prompt=PROMPT, few_shot_context=few_shot_context),
                                    batched=True, remove_columns=RENAMED_COLS, num_proc=4)




    # Prepare Evaluation
    test_dataloader = DataLoader(
        tokenized_test_ds,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=4
    )


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


    # ROUGE evaluation over Test dataset
    rouges = []
    #model.to(torch.float16)
    model.eval()
    for test_batch in tqdm(test_dataloader):
        with torch.no_grad():
            prompt = test_batch["input_ids"].to(DEVICE)
            suppress_tokens = [tokenizer.eos_token_id, tokenizer(" ")["input_ids"][-1]]  # 1437 is the space (blank_token_id)
            prompt_and_pred = model.generate(prompt, max_new_tokens=MAX_TARGET_LENGTH, do_sample=False, top_k=50, top_p=0.9, begin_suppress_tokens=suppress_tokens, stopping_criteria=stopping_criteria)#, do_sample=True, num_beams=5)
            predictions = prompt_and_pred[:, prompt.shape[1]:]

            # Compute rouge score
            decoded_labels = tokenizer.batch_decode(test_batch["labels"], skip_special_tokens=True)
            rouge_batch = compute_rouge_eval(predictions.cpu().detach().numpy(), decoded_labels, tokenizer)
            print(decoded_labels[0])
            print(rouge_batch)
        rouges.append(rouge_batch)

    rouges = pd.DataFrame(rouges)
    print(rouges)
    print(f"ROUGE scores for dataset {TEST_DS['path']}:\n\n{rouges.describe()}")


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint")
    parser.add_argument("--nb_test_examples", type=int)

    args = parser.parse_args()
    evaluate_summarization(args.model_checkpoint, args.nb_test_examples)