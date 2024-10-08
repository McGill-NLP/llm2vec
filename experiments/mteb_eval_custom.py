import argparse
from typing import Any
import mteb
import json
import torch

import numpy as np
from mteb.models.instructions import task_to_instruction
from mteb.models.text_formatting_utils import corpus_to_texts

from llm2vec import LLM2Vec

def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction


class LLM2VecWrapper:
    def __init__(self, model=None, task_to_instructions=None):

        self.task_to_instructions = task_to_instructions
        self.model = model

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else llm2vec_instruction(task_to_instruction(prompt_name))
            )
        else:
            instruction = ""

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        sentences = [["", sentence] for sentence in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    )
    parser.add_argument("--task_name", type=str, default="STS16")
    parser.add_argument(
        "--task_to_instructions_fp",
        type=str,
        default="test_configs/mteb/task_to_instructions.json",
    )
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    task_to_instructions = None
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)

    l2v_model = LLM2Vec.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    model = LLM2VecWrapper(model=l2v_model, task_to_instructions=task_to_instructions)
    tasks = mteb.get_tasks(tasks=[args.task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir)
