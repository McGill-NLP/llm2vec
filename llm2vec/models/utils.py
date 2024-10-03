import importlib.metadata
from packaging import version
from transformers.utils.import_utils import _is_package_available


def is_transformers_attn_greater_or_equal_4_43_1():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.43.1"
    )