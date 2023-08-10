from .default import MNISTCollator


def get_collator(name):
    return {
        "default": MNISTCollator(),
    }[name]
