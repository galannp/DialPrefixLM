# Imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM, PeftModel
import torch
from DialPrefixLM.utils import print_gpu_utilization

def build_model_tokenizer(model_checkpoint, add_pretrain_tokens=False, load_in_8bit=False, bnb_config=None, lora_config=None, device_map="auto"):
    # Build tokenizer
    pretrain_tokens = {
        "mask_token":"[MASK]",
        "additional_special_tokens":["[bopref]", "[eopref]"]
    } if add_pretrain_tokens else {}
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, **pretrain_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    # Build model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, load_in_8bit=load_in_8bit, quantization_config=bnb_config, device_map=device_map, torch_dtype=torch.bfloat16)
    if add_pretrain_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    
    if load_in_8bit or bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    print_gpu_utilization()
    print(model, tokenizer)
    return model, tokenizer


def save_model_tokenizer(model, tokenizer, save_path, is_lora=False, base_model_builder=None):
    # Dequantize, Merge LORA Adapters back into model weights and Save model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model
    torch.cuda.empty_cache()

    if is_lora:
        if base_model_builder is not None:
            save_path_merged = save_path + "_merged"

            # Load LoRA adapter and merge
            base_model = base_model_builder()
            model = PeftModel.from_pretrained(base_model, save_path)
            model = model.merge_and_unload()

            model.save_pretrained(save_path_merged, safe_serialization=True)
            tokenizer.save_pretrained(save_path_merged)
        else:
            raise ValueError("A `base_model_builder` must be specified when `is_lora` is set to True")
