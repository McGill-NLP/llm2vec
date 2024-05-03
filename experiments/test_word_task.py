import os
import sys
import logging
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, set_seed, HfArgumentParser
import torch
from datasets import load_dataset
import evaluate
import json
from tqdm import tqdm
from run_word_task import ModelForWordTask
from llm2vec import LLM2Vec


LABELS = {
    "conll2003": {
        "pos_tags": {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
                    'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
                    'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
                    'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
                    'WP': 44, 'WP$': 45, 'WRB': 46},
        "chunk_tags": {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
                    'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
                    'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22},
        "ner_tags": {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    }
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="custom", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--peft_addr", default=None, type=str, help="The dir address where adapter_model.bin is saved.")
    parser.add_argument("--cls_addr", default=None, type=str, help="The dir address where classifier is saved.")
    parser.add_argument("--bidirectional", default=True, type=str2bool)
    parser.add_argument("--merge_subwords", default=True, type=str2bool)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--classifier_dropout", default=0.1, type=float)
    parser.add_argument("--attn_implementation", default="sdpa", type=str, choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--torch_dtype", default=None, type=str, choices=["auto", "bfloat16", "float16", "float32"])

    parser.add_argument("--retroactive_labels", default="next_token", type=str, choices=["next_token", "same_token"])
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--task", default=None, type=str, choices=["pos_tags", "chunk_tags", "ner_tags"])
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=32, type=int)

    parser.add_argument("--config_file", default=None, type=str)

    args = parser.parse_args()

    if args.config_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        from pathlib import Path
        import json

        json_text = json.load(open(os.path.abspath(args.config_file)))
        argparse_dict = vars(args)
        argparse_dict.update(json_text)
        # args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args()

    path_to_check = args.peft_addr if args.peft_addr else args.model_name_or_path
    assert args.output_dir is not None, "If you want to evaluate a model, you have to provide the output_dir"
    os.makedirs(args.output_dir,exist_ok=True)

    set_seed(args.seed)
    
    tokenizer_kwargs = {}
    if "gpt" in args.model_name_or_path:
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_class == "custom":
        tokenizer.model_input_names.append("token_type_ids")

    if args.model_class == "auto":
        assert not args.merge_subwords

    assert args.dataset_name in LABELS and args.task in LABELS[args.dataset_name], f"LABELS[{args.dataset_name}][{args.task}] is not defined."

    config_kwargs = {
        "num_labels": len(LABELS[args.dataset_name][args.task]),
        "id2label": {i: lab for (lab, i) in LABELS[args.dataset_name][args.task].items()},
        "label2id": LABELS[args.dataset_name][args.task],
        "classifier_dropout": args.classifier_dropout
    }

    if args.model_class == "custom":
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
        else:
            raise ValueError('Invalid config loading')

        for k, v in config_kwargs.items():
            config.__setattr__(k, v)

        torch_dtype = (
            args.torch_dtype
            if args.torch_dtype in ["auto", None]
            else getattr(torch, args.torch_dtype)
        )
        l2v = LLM2Vec.from_pretrained(
            base_model_name_or_path=args.model_name_or_path,
            enable_bidirectional=args.bidirectional,
            peft_model_name_or_path=args.peft_addr,
            merge_peft=False,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
        )
        model = ModelForWordTask(
            model=l2v.model, 
            merge_subwords=args.merge_subwords, 
            config=config, 
            torch_dtype=torch_dtype,
            )

        classifier_path = os.path.join(args.cls_addr, "classifier.pt")
        if os.path.exists(classifier_path):
            print(f"Loading classifier from {classifier_path}")
            model.classifier = torch.load(classifier_path)
        else:
            raise ValueError("classifier does not exist in", classifier_path)

    elif args.model_class == "auto":
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, 
                                                                num_labels=len(LABELS[args.dataset_name][args.task]),
                                                                id2label={i: lab for (lab, i) in LABELS[args.dataset_name][args.task].items()},
                                                                label2id=LABELS[args.dataset_name][args.task])
    else:
        raise ValueError(f"{args.model_class} is not implemented. Only auto and custom model_class options are valid.")

    model = model.cuda()

    raw_datasets = load_dataset(
            args.dataset_name, split="test"
        )
    
    def tokenize_and_align_labels(examples):
        task = args.task
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=args.max_seq_length, return_tensors="pt")

        labels = []
        words = []
        for i, label in enumerate(examples[task]):
            if args.retroactive_labels in ["same_token"]:
                # if args.retroactive_labels == "next_word":
                #     label = label[1:] + [-100]
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
                word_ids = [-1 if w is None else w for w in word_ids]
                words.append(word_ids)
            elif args.retroactive_labels == "next_token":
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                label_ids.append(-100)
                labels.append(label_ids[1:])
                word_ids = word_ids[1:] + [None]
                word_ids = [-1 if w is None else w for w in word_ids]
                words.append(word_ids)
            else:
                raise ValueError(f"retroactive_labels {args.retroactive_labels} is not implemented.")

        tokenized_inputs["labels"] = torch.tensor(labels)
        if args.model_class == "custom":
            tokenized_inputs["token_type_ids"] = words
        return tokenized_inputs
    
    tokenized_dataset = raw_datasets.map(tokenize_and_align_labels, batched=True, remove_columns=list(LABELS[args.dataset_name].keys())+["tokens", "id"])
    with torch.no_grad():
        predictions = None
        labels = None
        for batch_begin in tqdm(torch.arange(0, len(tokenized_dataset), args.batch_size)):
            features = {
                "input_ids": torch.tensor(tokenized_dataset[batch_begin:batch_begin+args.batch_size]["input_ids"]).to(model.device),
                "attention_mask": torch.tensor(tokenized_dataset[batch_begin:batch_begin+args.batch_size]["attention_mask"]).to(model.device)
            }
            if "token_type_ids" in tokenized_dataset[batch_begin:batch_begin+args.batch_size]:
                features["token_type_ids"] = torch.tensor(tokenized_dataset[batch_begin:batch_begin+args.batch_size]["token_type_ids"]).to(model.device)
                                                
            labs = torch.tensor(tokenized_dataset[batch_begin:batch_begin+args.batch_size]["labels"])

            logits = model(**features).logits
            preds = torch.argmax(logits, dim=-1)
            if predictions is None:
                predictions = preds
                labels = labs
            else:
                predictions = torch.concatenate((predictions, preds))
                labels = torch.concatenate((labels, labs))

    precision_metric = evaluate.load("precision")
    metrics = precision_metric.compute(references=labels[labels != -100], predictions=predictions[labels != -100], average="micro")  
    
    with open(os.path.join(args.output_dir, "result_summary.json"), "w") as f:
        json.dump(metrics, f)
    print(metrics)