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

def evaluate_next_utt_gen(MODELS, NB_TEST_EXAMPLES):
    # Constants
    #TEST_DS = {"path": "TanveerAman/AMI-Corpus-Text-Summarization", "split": "validation", "extract_cols": ["Dialogue", "Summaries"]}
    #TEST_DS = {"path": "knkarthick/dialogsum", "split": "validation", "extract_cols": ["dialogue"]}
    TEST_DS = {"path": "ccdv/mediasum", "split": "validation", "extract_cols": ["document"]}
    RENAMED_COLS = ["dialogue"]

    DEVICE = "cuda"
    BATCH_SIZE = 1
    CONTEXT_LENGTH = 2048
    MAX_TARGET_LENGTH = 64


    # Load tested model and tokenizer
    eval_models = [build_model_tokenizer(model_path, device_map=f"{DEVICE}:{id_gpu}")[0] for id_gpu, model_path in enumerate(MODELS)]
    tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
    print_gpu_utilization()

    def preprocess_for_next_utt_gen(examples):
        def extract_context_and_next_utt(dial):
            #dial = dial.replace("#Person1#", "Sam").replace("#Person2#", "Thomas")
            dial = dial[:min(len(dial), 7000)]
            utterances = dial.split("\n")[:-1]
            contexts = []
            next_utts = []
            for i in range(5, len(utterances)):
                speaker_i, utt_i = utterances[i].split(":", maxsplit=1)
                contexts.append("\n".join(utterances[:i]) + "\n" + speaker_i + ":")
                next_utts.append(utt_i)
            return contexts, next_utts

        all_contexts = []
        all_next_utts = []
        for ex in examples["dialogue"]:
            contexts, next_utts = extract_context_and_next_utt(ex)
            all_contexts.extend(contexts)
            all_next_utts.extend(next_utts)

        # Preprocess input
        tk_examples = tokenizer(all_contexts, truncation=False)

        # Preprocess labels
        tk_examples["labels"] = tokenizer(all_next_utts, truncation=False)["input_ids"]

        return tk_examples

    # Load test Dataset
    test_ds = build_single_dataset(**TEST_DS, renamed_cols=RENAMED_COLS)
    if NB_TEST_EXAMPLES is not None:
        test_ds = test_ds.shuffle().select(range(min(NB_TEST_EXAMPLES, len(test_ds))))
    print(test_ds)

    # Preprocess test Dataset
    tokenized_test_ds = test_ds.map(lambda examples: preprocess_for_next_utt_gen(examples), batched=True, remove_columns=RENAMED_COLS, num_proc=4)


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
    from collections import defaultdict

    rouges = defaultdict(lambda: [])
    [eval_model.eval() for eval_model in eval_models]
    for test_batch in tqdm(test_dataloader):
        with torch.no_grad():
            context = test_batch["input_ids"]
            suppress_tokens = [tokenizer.eos_token_id, tokenizer(" ")["input_ids"][-1]]

            print(tokenizer.decode(context[0]).replace("\n", "\n\n"))
            decoded_labels = tokenizer.batch_decode(test_batch["labels"], skip_special_tokens=True)
            print(f"Groundtruth for next utterance:\n{decoded_labels[0]}\n")

            for id_gpu, (model_name, eval_model) in enumerate(zip(MODELS, eval_models)):
                context_and_pred = eval_model.generate(context.to(f"{DEVICE}:{id_gpu}"), max_new_tokens=MAX_TARGET_LENGTH, do_sample=True, top_k=50, top_p=0.9, begin_suppress_tokens=suppress_tokens, stopping_criteria=stopping_criteria)#, do_sample=True, num_beams=5)
                predictions = context_and_pred[:, context.shape[1]:]
                #print(predictions.shape[1], test_batch["labels"].shape[1])
                # Compute rouge score
                print(f"Prediction for {model_name}:")
                rouge_batch = compute_rouge_eval(predictions.cpu().detach().numpy(), decoded_labels, tokenizer)
                print(rouge_batch, "\n\n")
                rouges[model_name].append(rouge_batch)
    print_gpu_utilization()

    for model_name, rouge_stats in rouges.items():
        rouge_stats = pd.DataFrame(rouge_stats)
        print(rouge_stats)
        print(f"ROUGE scores for model {model_name} on dataset {TEST_DS['path']}:\n\n{rouge_stats.describe()}")


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--nb_test_examples", type=int)

    args = parser.parse_args()
    evaluate_next_utt_gen(args.models, args.nb_test_examples)