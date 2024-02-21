import evaluate
import numpy as np

rouge = evaluate.load("rouge")

def compute_rouge_eval(preds, labels, tokenizer):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(decoded_preds[0])

    result = rouge.compute(predictions=decoded_preds, references=labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}