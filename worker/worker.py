import sys
import torch


def main() -> int:
    print("torch", torch.__version__)
    print("cude", torch.cuda.is_available())
    print("threads", torch.get_num_threads())

    return 0


if __name__ == "__main__":
    main()
