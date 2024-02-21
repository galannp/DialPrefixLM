## Setup

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
#os.environ["TOKENIZERS_PARALLELISM"] = "true"


## GPU Info

# Imports
from DialPrefixLM.utils import print_gpu_available, print_gpu_utilization

print_gpu_available()
print_gpu_utilization()




def pretrain(MODEL_CHECKPOINT, DATASETS, MODEL_SAVE_NAME, FCM, IS_DIAL_DS, BALANCED_MIXTURE, MAX_SAMPLES_PER_DS, RESUME_TRAINING, WANDB_IDS):
    """
        FCM parameter:
            - False: Causal Language Modeling objective (no masking)
            - True: Causal Language Modeling with Dialogue Noise Injection
    """

    ## Pretraining Setting
    DS_PARAMS = [
        {"path": "wikipedia", "name": "20220301.en", "split": "train", "extract_cols": ["text"]},
        {"path": "ccdv/mediasum", "split": "train", "extract_cols": ["document"]},
        {"path": "samsum", "split": "train", "extract_cols": ["dialogue"]},
        #{"path": "Salesforce/dialogstudio", "name": "SODA", "split": "train", "extract_cols": extract_dialogues_soda},
        {"path": "knkarthick/dialogsum", "split": "train", "extract_cols": ["dialogue"]},
        {"path": "TanveerAman/AMI-Corpus-Text-Summarization", "split": "train", "extract_cols": ["Dialogue"]}
    ]
    TRAIN_DS = [ds_params for ds_params in DS_PARAMS if ds_params["path"] in DATASETS]
    TRAIN_COL = ["dialogue"]

    IS_LORA = True
    IS_8BIT = False
    IS_4BIT = False

    MODEL_SAVE_PATH = f"post_train_models/{MODEL_SAVE_NAME}"


    ## Quantization + LORA + ZeRO-2

    # Imports
    from DialPrefixLM.model import build_model_tokenizer
    from transformers import BitsAndBytesConfig
    from accelerate import Accelerator
    from peft import LoraConfig
    import torch

    # Constants
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if IS_4BIT else None

    LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
        ],
        lora_dropout=0.1,
        bias="none",
        modules_to_save = ["lm_head", "embed_tokens"] if FCM else [],
        task_type="CAUSAL_LM"
    ) if IS_LORA else None

    """target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],"""
    """target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
        ],"""

    accelerator = Accelerator()
    current_device = accelerator.process_index
    device_map = {"": current_device}
    model, tokenizer = build_model_tokenizer(MODEL_CHECKPOINT, add_pretrain_tokens=FCM, load_in_8bit=IS_8BIT, bnb_config=BNB_CONFIG, lora_config=LORA_CONFIG, device_map=device_map)



    ## Preprocessing

    # Imports
    from DialPrefixLM.preprocess.build_dialogue_datasets import build_single_dataset, extract_dialogues_soda
    from DialPrefixLM.preprocess.inject_noise_to_dialogue import inject_dialogue_noise
    from DialPrefixLM.preprocess.inject_noise_to_text import inject_text_noise
    from datasets import concatenate_datasets, load_from_disk
    from time import time
    import numpy as np

    # Constants

    CONTEXT_LENGTH = 2048

    CACHED_DS_PATH = f"post_train_ds/cached_ds_{MODEL_SAVE_NAME}"


    def preprocess(examples, context_length, inject_dial_noise=False):
        # TODO: change name of variables and keep consistancy within the notebook (bopref) doesn't mean anything anymore
        # TODO: add eopref and bopref when one span is split between two samples from the batch
        if inject_dial_noise:
            # To fit to the chosen LLM
            noise_fun = inject_dialogue_noise if IS_DIAL_DS else inject_text_noise
            spans, idx_noised = noise_fun(examples["dialogue"], bos_token=tokenizer.bos_token)
            spans_ids = tokenizer(spans, truncation=False, add_special_tokens=False)["input_ids"]
            spans_mask = [[span_type] * len(span_tk) for span_tk, span_type in zip(spans_ids, idx_noised)]
            attention_mask = [[1] * len(span_tk) for span_tk in spans_ids]

            tk_examples = {
                "input_ids": spans_ids,
                "attention_mask": attention_mask,
                "prefix_mask": spans_mask
            }
        else:
            examples["dialogue"] = [tokenizer.bos_token + dial for dial in examples["dialogue"]]
            tk_examples = tokenizer(examples["dialogue"], truncation=False, add_special_tokens=False)

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


    # Build/Load Preprocessed Datasets
    if accelerator.is_main_process:
        if os.path.isdir(CACHED_DS_PATH):
            tokenized_post_training_ds = load_from_disk(CACHED_DS_PATH)
            print(f"Length of loaded dataset from disk: {len(tokenized_post_training_ds)}")
        else:
            dataset_list = []
            for ds_params in TRAIN_DS:
                print(f"\nProcessing and Tokenizing {ds_params['path']} dataset")
                ds = build_single_dataset(**ds_params, renamed_cols=TRAIN_COL)

                t = time()
                ds = ds.shuffle()
                if MAX_SAMPLES_PER_DS is not None:
                    ds = ds.select(range(min(len(ds), MAX_SAMPLES_PER_DS)))

                preproc = lambda examples: preprocess(examples, CONTEXT_LENGTH, inject_dial_noise=FCM)
                tokenized_ds = ds.map(preproc, batched=True, remove_columns=TRAIN_COL, num_proc=4)

                print(f"Total length of tokenized dataset: {len(tokenized_ds)}")
                print(f"Time spent preprocessing noise to the dataset: {time() - t}")

                dataset_list.append(tokenized_ds)

            if BALANCED_MIXTURE:
                min_ds_size = min([len(ds) for ds in dataset_list])
                print(f"Balanced mixture of datasets: selecting only {min_ds_size} samples from each dataset")
                dataset_list = [ds.shuffle().select(range(min_ds_size)) for ds in dataset_list]

            # Stats
            print("Mixture of datasets contains:")
            nb_sample_per_ds = np.array([len(ds) for ds in dataset_list])
            ratio_per_ds = 100 * nb_sample_per_ds / sum(nb_sample_per_ds)
            [print(f"{ratio_per_ds[i]}% of dataset {TRAIN_DS[i]['path']}") for i in range(len(dataset_list))]

            tokenized_post_training_ds = concatenate_datasets(dataset_list).shuffle()
            tokenized_post_training_ds.save_to_disk(CACHED_DS_PATH)

    else:
        accelerator.wait_for_everyone()


    if not accelerator.is_main_process:
        tokenized_post_training_ds = load_from_disk(CACHED_DS_PATH)
        print(f"Length of loaded dataset from disk: {len(tokenized_post_training_ds)}")
    else:
        accelerator.wait_for_everyone()

    # Eval
    """tokenized_post_train_ds = tokenized_post_training_ds.train_test_split(test_size=2500, shuffle=True)
    tokenized_train_ds, tokenized_eval_ds = tokenized_post_train_ds["train"], tokenized_post_train_ds["test"]
    print(len(tokenized_train_ds), len(tokenized_eval_ds))"""

    """tokenized_train_ds = tokenized_post_training_ds
    EVAL_DS = {"path": "ccdv/mediasum", "split": "validation", "extract_cols": ["document"]}
    ds_eval = build_single_dataset(**EVAL_DS, renamed_cols=TRAIN_COL)
    preproc = lambda examples: preprocess(examples, CONTEXT_LENGTH, inject_dial_noise=FCM)
    tokenized_eval_ds = ds_eval.map(preproc, batched=True, remove_columns=TRAIN_COL, num_proc=4)
    tokenized_eval_ds = tokenized_eval_ds.shuffle().select(range(2500))"""

    tokenized_train_ds = tokenized_post_training_ds
    EVAL_DS = TRAIN_DS[0]
    EVAL_DS["split"] = "validation"
    def preprocess_eval(examples, context_length):
        examples["dialogue"] = [tokenizer.bos_token + dial for dial in examples["dialogue"]]
        tk_examples = tokenizer(examples["dialogue"], truncation=False, add_special_tokens=False)
        tk_examples["prefix_mask"] = [[0] * len(a) for a in tk_examples["input_ids"]]

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
    ds_eval = build_single_dataset(**EVAL_DS, renamed_cols=TRAIN_COL)
    ds_eval = ds_eval.shuffle(seed=42).select(range(3000))
    if FCM:
        tokenized_eval_ds = ds_eval.map(lambda examples: preprocess_eval(examples, CONTEXT_LENGTH),
                                                batched=True, remove_columns=TRAIN_COL, num_proc=4)
    else:
        tokenized_eval_ds = ds_eval.map(lambda examples: preprocess(examples, CONTEXT_LENGTH, inject_dial_noise=FCM),
                                                batched=True, remove_columns=TRAIN_COL, num_proc=4)



    ## Post Training

    # Imports
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from DialPrefixLM.preprocess.data_collators import DataCollatorForPrefixLM
    from DialPrefixLM.utils import print_summary
    from DialPrefixLM.model import save_model_tokenizer

    # Constants
    TRAIN_ARGS=TrainingArguments(
        per_device_train_batch_size=8, #4,
        gradient_accumulation_steps=8, #16,
        per_device_eval_batch_size=8, #8,
        warmup_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        #fp16=True,
        logging_steps=1,
        output_dir=f"outputs/{MODEL_SAVE_NAME}",
        #optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        label_names=["labels"],
        remove_unused_columns=False,
    ) 

    if FCM:
        data_collator = DataCollatorForPrefixLM(tokenizer)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    ## Train with Trainer

    """# Examine dataloaders
    print("\n\n\n\n\n")
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        tokenized_train_ds,
        shuffle=True,
        batch_size=4,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=1
    )

    for a in train_dataloader:
        #print(tokenizer.decode(a["input_ids"][0]).replace("\n", "\n\n").replace("</s>", "</s>\n\n\n"))
        print(tokenizer.decode(a["input_ids"][0]))
        print("\n\n")
        labels = []
        for l in a["labels"][0]:
            if l != -100:
                labels.append(l)
            else:
                labels.append(tokenizer.pad_token_id)
        #print(tokenizer.decode(labels).replace("\n", "\n\n"))
        print(tokenizer.decode(labels))
        break"""

    import wandb

    if RESUME_TRAINING:
        wandb_id = WANDB_IDS[current_device]
    else:
        wandb_id = wandb.util.generate_id()
    print(f"GPU {current_device} has wandb_id: {wandb_id}")
    wandb.init(
        id=wandb_id,
        resume="allow",
        project="pretrain_dial_fcm_phi_debug",
        group=MODEL_SAVE_NAME
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        args=TRAIN_ARGS,
        data_collator=data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train(resume_from_checkpoint=RESUME_TRAINING)
    print_summary(result)
    wandb.finish()

    if accelerator.is_main_process:
        # Save model
        base_model_builder = lambda: build_model_tokenizer(MODEL_CHECKPOINT, add_pretrain_tokens=FCM)[0]
        save_model_tokenizer(trainer.model, tokenizer, MODEL_SAVE_PATH, is_lora=IS_LORA, base_model_builder=base_model_builder)


    # Llama adptation:
    # - warnings
    # Evaluation
    # Choose noise for FCM

    # How to noise short utterances.
    # 1. the length of a span could depend on the length of it in terms of words (not robust because then two much small utterances in the same span). Then other option (1 long utterance = 2 small utterances)
    # 2. how to avoid masks at the end of an utterance?
    # 3. set a limit max to the size of the noised span (max characters)


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint")
    parser.add_argument("model_save_name")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--fcm", action="store_true")
    parser.add_argument("--dial_ds", action="store_true")
    parser.add_argument("--balanced_mixture", action="store_true")
    parser.add_argument("--max_samples_per_ds", type=int)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--wandb_ids", nargs="+")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--deepspeed")

    args = parser.parse_args()
    pretrain(args.model_checkpoint, args.datasets, args.model_save_name,
            args.fcm, args.dial_ds, args.balanced_mixture,
            args.max_samples_per_ds, args.resume_training, args.wandb_ids)
