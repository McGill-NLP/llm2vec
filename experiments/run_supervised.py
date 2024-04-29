import argparse
import logging
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    LlamaConfig,
    MistralConfig,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss
from llm2vec.experiment_utils import (
    generate_experiment_id,
    log_commandline_args,
    set_seed,
    str2bool,
    prepare_model_args,
)
from tqdm import tqdm

transformers.logging.set_verbosity_error()


@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(model, text, pooling_mode=model.pooling_mode)
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class LLM2VecSupervisedTrainer(Trainer):

    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        loss = self.loss_function(q_reps, d_reps, d_reps_neg)

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def get_train_dataloader(self) -> DataLoader:
        # Copying most of the code from the parent class, changing the sampler to SequentialSampler
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # Changing from random sampler to sequential sampler
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + text.strip()
            + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"

    return text

def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model

# TODO: Parse these into JSON, organize same way as MNTP
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="t5-base")
parser.add_argument("--peft_addr", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="E5")
parser.add_argument(
    "--dataset_file_path",
    type=str,
    default="cache/echo-data",
)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--dev_batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=384, type=int)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument("--warmup_steps", default=0, type=int)
parser.add_argument("--checkpoint_save_steps", default=10000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--find_unused_parameters", default=False, type=str2bool)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--pooling_mode", default="mean", type=str)
parser.add_argument("--checkpoint_save_total_limit", default=0, type=int)
parser.add_argument("--experiment_id", default=None, type=str)
parser.add_argument("--grad_accumulation_steps", default=1, type=int)
parser.add_argument("--lora_r", default=8, type=int)
parser.add_argument("--lora_dropout", default=0.05, type=float)
parser.add_argument("--num_cpu_workers", default=4, type=int)
parser.add_argument("--bidirectional", default=False, type=str2bool)
parser.add_argument("--stop_after_n_steps", default=None, type=int)
parser.add_argument("--fp16", default=False, type=str2bool)
parser.add_argument("--bf16", default=False, type=str2bool)
parser.add_argument("--flash_attention_2", default=False, type=str2bool)
parser.add_argument("--load_in_8bit", default=False, type=str2bool)
parser.add_argument("--load_in_4bit", default=False, type=str2bool)
parser.add_argument("--amp", default=False, type=str2bool)
parser.add_argument("--deepspeed", default=None, type=str)
parser.add_argument("--gradient_checkpointing", default=False, type=str2bool)
parser.add_argument("--loss_cls", default="HardNegativeNLLLoss", type=str)
parser.add_argument("--loss_scale", default=50.0, type=float)

args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__, log_level="INFO")

    if args.find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(args)

    # if accelerator.is_main_process:
    log_commandline_args(args, logger.info)

    if args.deepspeed:
        assert (
            False
        ), "DeepSpeed is not implemented yet. There will be a problem with model saving and loading."

    if args.flash_attention_2:
        assert args.fp16 or args.bf16 or args.load_in_8bit or args.load_in_4bit

    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": False}

    if args.experiment_id is not None:
        experiment_id = args.experiment_id
    else:
        experiment_id = generate_experiment_id(
            name=args.dataset_name,
            split="train",
            model_name=(
                args.model_name
                if "/" not in args.model_name
                else args.model_name.split("/")[-1]
            ),
            pooling_mode=args.pooling_mode,
            train_batch_size=args.train_batch_size
            * accelerator.num_processes
            * args.grad_accumulation_steps,
            max_seq_length=args.max_seq_length,
            bidirectional=args.bidirectional,
            epochs=args.epochs,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            lr=args.lr,
            lora_r=args.lora_r,
        )

    model_save_path = f"{args.output_dir}/{experiment_id}"

    # TODO: can also pass separator arg here
    train_dataset = load_dataset(
        args.dataset_name,
        split="train",
        file_path=args.dataset_file_path,
        effective_batch_size=args.train_batch_size * accelerator.num_processes,
    )

    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    model_args = prepare_model_args(
        bf16=args.bf16,
        fp16=args.fp16,
        flash_attention_2=args.flash_attention_2,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.model_name,
        enable_bidirectional=args.bidirectional,
        peft_model_name_or_path=args.peft_addr,
        merge_peft=True,
        pooling_mode=args.pooling_mode,
        max_length=args.max_seq_length,
        **model_args,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=args.lora_dropout,
    )

    tokenizer = model.tokenizer

    train_loss = load_loss(args.loss_cls, scale=args.loss_scale)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=args.epochs,
        seed=args.seed,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        logging_dir=model_save_path + "/logs",
        logging_steps=50,
        save_steps=args.checkpoint_save_steps,
        save_total_limit=args.checkpoint_save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=False,
        save_only_model=True,
        fp16=args.amp,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=args.find_unused_parameters,
    )

    data_collator = DefaultCollator(model)

    trainer = LLM2VecSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
    )

    if args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(args.stop_after_n_steps))

    trainer.train()