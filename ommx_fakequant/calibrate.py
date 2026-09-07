"""calibrate.py — calibration-data loader for weight quantization.

Provides get_calib_dataset(), which samples fixed-length token windows from a calibration
corpus (Pile / C4 / WikiText) used by the OMMX weight quantizer in wq_eval.py.
"""
import torch
from datasets import load_dataset


def get_calib_dataset(tokenizer, nsamples=128, seqlen=2048, dataset_name="pile-10k"):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(dataset["text"])
    elif dataset_name == "c4":
        # stream C4 (avoid full download); accumulate enough chars for nsamples crops
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts, total, need = [], 0, seqlen * nsamples * 8
        for ex in ds:
            t = ex.get("text", "")
            texts.append(t); total += len(t)
            if total >= need:
                break
        text = "\n\n".join(texts)
    else:
        # default corpus
        dataset = load_dataset("NeelNanda/pile-10k", split="train")
        text = "\n\n".join(dataset["text"])

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids

    # Tokenizers that do NOT auto-prepend BOS (e.g. EXAONE-4.0, add_bos_token=False) need an
    # attention-sink BOS at the start of every crop -- without it these random mid-sequence
    # windows are the same degenerate regime that collapsed wikitext PPL 57->30 (wq_eval.py
    # wikitext_ppl). Calibration crops are just as BOS-less, so the per-channel Hessian
    # statistics the error-feedback solver relies on were built on collapsed activations.
    # add_bos=True models (Llama/Qwen/Mistral) keep the original crop path byte-identical.
    needs_bos = (getattr(tokenizer, "add_bos_token", False) is False
                 and tokenizer.bos_token_id is not None)
    crop_len = seqlen - 1 if needs_bos else seqlen
    calib_data = []
    for _ in range(nsamples):
        i = torch.randint(0, input_ids.shape[1] - crop_len - 1, (1,)).item()
        j = i + crop_len
        crop = input_ids[:, i:j]
        if needs_bos:
            bos = torch.tensor([[tokenizer.bos_token_id]])
            crop = torch.cat([bos, crop], dim=1)
        calib_data.append(crop)
    return torch.cat(calib_data, dim=0)
