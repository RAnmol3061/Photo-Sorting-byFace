import logging
import sys
from pathlib import Path


def validate_cli_dir(cli_flag_value: Path | None, default_value: str | None) -> Path:
    """Resolve and validate

    Method is used to validate whether cli argument is resolves to real path. If not, then default value is
    checked if resolves to real path. If both values do not resolve then exit from the program

    Args:
        cli_flag_value (Path | None): Value provided as cli argument. If no cli argument provided, value becomes none
        default_value (str | None): Value provided as default. If no default provided, value assigned none

    Returns:
        A resolved, existing

    Raises:
        FileNotFoundError: If provided cli arguments & default values don't resolv
    """
    if cli_flag_value is not None:
        try:
            return cli_flag_value.resolve(strict=True)
        except FileNotFoundError:
            logging.warning("Invalid CLI Argument, moving to default values(env)")
            print("Invalid CLI Argument, moving to default values(env)")
    if default_value is not None:
        try:
            return Path(default_value).resolve(strict=True)
        except FileNotFoundError:
            logging.critical("Invalid Default value in .env file. Exiting")
            print("Invalid Default value. Exiting")
            sys.exit(1)
    logging.critical("Exiting: Due to no cli & default arguments")
    print("Error: No cli argument and empty default. Exiting")
    sys.exit(1)
