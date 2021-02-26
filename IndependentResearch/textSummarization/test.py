import argparse

"""
This is some function that has no impact on how to do this whatsover.
"""


def main():
    """
    This is another comment that is just perhaps a little too long for its own good.
    """
    a = "b"
    print("Hello World!")
    print("This is something")
    some_func()
    a = "c"


def some_func():
    print("YES!")
    b = list(map(lambda x: x ^ 2, [1, 2, -3]))
    print(b)


if __name__ == "__main__":
    main()
