"""python -m tui [--ports 9201,9202,9203,9204]"""
import argparse

from tui.app import Pouwtop


def main() -> None:
    ap = argparse.ArgumentParser(prog="pouwtop", description="монітор мережі pouwchain")
    ap.add_argument("--ports", default="9201,9202,9203,9204", help="порти HTTP API вузлів")
    args = ap.parse_args()
    Pouwtop([int(p) for p in args.ports.split(",") if p]).run()


if __name__ == "__main__":
    main()
