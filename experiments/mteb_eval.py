import argparse
import mteb
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
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

    model_kwargs = {}
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)
        model_kwargs["task_to_instructions"] = task_to_instructions

    model = mteb.get_model(args.model_name, **model_kwargs)
    tasks = mteb.get_tasks(tasks=[args.task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir)
