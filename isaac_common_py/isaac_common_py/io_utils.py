import sys


def print_green(text: str):
    """ Print text in green. """
    print(f"\033[32m{text}\033[0m")


def print_yellow(text: str):
    """ Print text in yellow. """
    print(f"\033[33m{text}\033[0m")


def print_blue(text: str):
    """ Print text in blue. """
    print(f"\033[34m{text}\033[0m")


def print_gray(text: str):
    """ Print text in gray. """
    print(f"\033[90m{text}\033[0m")


def print_red(text: str):
    """ Print text in red. """
    print(f"\033[91m{text}\033[0m")


def delete_last_lines_in_stdout(n: int):
    """ Delete the last n lines in stdout. """
    sys.stdout.write("\033[F\033[K" * n)
