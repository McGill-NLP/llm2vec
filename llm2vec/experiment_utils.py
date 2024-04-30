import re


def generate_experiment_id(
    name,
    split,
    model_name,
    pooling_mode,
    train_batch_size,
    max_seq_length,
    bidirectional,
    epochs,
    seed,
    warmup_steps,
    lr,
    lora_r,
):
    experiment_id = name + "_" + split

    if isinstance(model_name, str):
        experiment_id += f"_m-{model_name}"
    if isinstance(pooling_mode, str):
        experiment_id += f"_p-{pooling_mode}"
    if isinstance(train_batch_size, int):
        experiment_id += f"_b-{train_batch_size}"
    if isinstance(max_seq_length, int):
        experiment_id += f"_l-{max_seq_length}"
    if isinstance(bidirectional, bool):
        experiment_id += f"_bidirectional-{bidirectional}"
    if isinstance(epochs, int):
        experiment_id += f"_e-{epochs}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(warmup_steps, int):
        experiment_id += f"_w-{warmup_steps}"
    if isinstance(lr, float):
        experiment_id += f"_lr-{lr}"
    if isinstance(lora_r, int):
        experiment_id += f"_lora_r-{lora_r}"

    return experiment_id

def parse_experiment_id(experiment_id):
    """
    Parses experiment identifier into key-value pairs.

    Args:
        experiment_id (str): Unique experiment identifier to parse.

    Returns:
        dict: Dictionary containing the parsed key-value pairs.
    """
    regex, post_regex = "", ""
    if "/" in experiment_id:
        regex = "([A-Za-z0-9-_./]*)/"
        post_regex = "/([A-Za-z0-9-_./]*)"
    regex += "([A-Za-z0-9-_.]+)"
    regex += "_m-([A-Z-a-z0-9-_.]+)"
    regex += "_p-([A-Z-a-z0-9-_.]+)"
    regex += "_b-(\d+)"
    regex += "_l-(\d+)"
    regex += "_bidirectional-([A-Z-a-z0-9-_.]+)"
    regex += "_e-(\d+)"
    regex += "_s-(\d+)"
    regex += "_w-(\d+)"
    regex += "_lr-([A-Z-a-z0-9-_.]+)"
    regex += "_lora_r-(\d+)"
    regex += post_regex

    parts = re.match(regex, experiment_id).groups()
    if post_regex != "":
        parts = parts[1:-1]

    result = {
        "name": parts[0],
        "model_name_or_path": parts[1],
        "pooling_mode": parts[2],
        "train_batch_size": int(parts[3]),
        "max_seq_length": int(parts[4]),
        "bidirectional": parts[5] == "True",
        "epochs": int(parts[6]),
        "seed": int(parts[7]),
        "warmup_steps": int(parts[8]),
        "lr": float(parts[9]),
        "lora_r": int(parts[10]),
    }

    return result
