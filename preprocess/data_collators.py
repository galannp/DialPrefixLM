from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Mapping

@dataclass
class DataCollatorForPrefixLM:
    """Data Collator for Prefix Language Modeling (PreLM) and Masked Prefix Language Modeling (MPreLM) objectives
    This implementation has only been designed for pytorch
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        prefix_mask = []
        for example in examples:
            prefix_mask.append(example["prefix_mask"])
            del example["prefix_mask"]
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        # Pad the prefix mask to the same size as the input
        padded_seq_len = batch["input_ids"].shape[1]
        prefix_mask_padded = torch.tensor([[0] * (padded_seq_len - len(pr_mask)) + pr_mask for pr_mask in prefix_mask])
        prefix_mask_padded_bool = prefix_mask_padded.bool()

        # Define the labels
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        labels[prefix_mask_padded_bool] = -100
        batch["labels"] = labels

        return batch
