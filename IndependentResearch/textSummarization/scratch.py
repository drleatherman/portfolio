#!/usr/bin/env python
import argparse


def main(args):
    """
    This is some doc
    """
    print(args)


def sub_function():
    """
    Here is some doc about this sub function
    """
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--type", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
