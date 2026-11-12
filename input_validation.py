import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_cli_dir(
    cli_flag_value: Path | None, default_value: str | None
) -> Path | bool:
    """Checks if provided args resolve, if not returns False.

    Takes two values. If cli_flag_value does not resolve or not exist
    falls back to default_value. If default value also does not resolve
    or not exist (none) then finally returns False. Otherwise returns resolved
    Path. Here resolved means path.resolve(strict=True)

    Args:
        cli_flag_value (Path | None): Value provided as cli argument.
                                      If no cli argument provided, value -> none

        default_value (str | None): Value provided as default. If
                                    no default provided, value -> none
    Returns:
        A valid path object or false boolean
    """
    if cli_flag_value is not None:
        try:
            return cli_flag_value.resolve(strict=True)
        except FileNotFoundError:
            logger.exception("Invalid CLI Argument, moving to default values (env)")
            print("Invalid CLI Argument, moving to default values(env)")
    if default_value is not None:
        try:
            return Path(default_value).resolve(strict=True)
        except FileNotFoundError:
            logger.critical("Invalid Default value in .env file. Exiting")
            print("Invalid Default value. Exiting")
            return False
    logger.critical("Exiting: Due to no cli & default arguments")
    print("Error: No cli argument and empty default. Exiting")
    return False
