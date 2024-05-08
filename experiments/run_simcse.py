import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss

from tqdm import tqdm

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )


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
                # TODO: Add prepare_for_tokenization here similar to supervised training and see if it impacts performance
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


class SimCSETrainer(Trainer):

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

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
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

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    train_dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
    )

    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer

    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    data_collator = DefaultCollator(model)

    trainer = SimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()
