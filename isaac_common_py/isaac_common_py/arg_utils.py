import argparse


def str_to_bool(value):
    """Convert string representation of boolean to actual boolean.

    Supports: true, false, True, False, 1, 0, yes, no, on, off
    Also treats --flag (without value) as true
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

    # Handle case where --flag is passed without a value (None)
    if value is None:
        return True

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {type(value)}")
