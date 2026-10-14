import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
os.environ["TORCH_EXTENSIONS_DIR"] = ""

from llmtuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
