"""
Utilities to validate configuration signatures and fetch configuration data.

This module exposes a single orchestration function:
    handle_config_file(file_name: str, sign: str) -> Optional[dict]

It also implements small, pure helpers to keep concerns separate and easy to test.
"""

import logging
from typing import Callable, Mapping, Optional


# Module-level logger for instrumentation of operations.
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid "No handler found" warnings if the application does not configure logging.
    logger.addHandler(logging.NullHandler())


# Provide fallbacks for external globals to satisfy linters/runtime when not injected.
try:
    TRUSTED_CONFIG_SIGNATURES  # type: ignore[name-defined]
except NameError:
    TRUSTED_CONFIG_SIGNATURES = {}

try:
    fetch_config_data  # type: ignore[name-defined]
except NameError:
    def fetch_config_data(file_name: str):
        raise IOError(f"Configuration file {file_name} does not exist")


def _verify_signature(
    file_name: str,
    sign: str,
    trusted_signatures: Mapping[str, str],
) -> bool:
    """
    Pure function: Validate that the provided signature matches the trusted one.

    Raises:
        IOError: If the file name is not known in trusted_signatures.
    """
    logger.debug("Verifying signature for '%s'.", file_name)

    if file_name not in trusted_signatures:
        logger.error("Configuration file '%s' not found in trusted signatures.", file_name)
        raise IOError(f"Configuration file {file_name} does not exist")

    expected = trusted_signatures[file_name]
    if sign == expected:
        logger.info("Signature verified for '%s'.", file_name)
        return True

    logger.warning("Signature verification failed for '%s'.", file_name)
    return False


def _fetch_configuration(file_name: str, fetcher: Callable[[str], dict]) -> dict:
    """
    Pure function: Return configuration data using the supplied fetcher callable.

    Propagates:
        IOError from the fetcher if the configuration cannot be located.
    """
    logger.info("Fetching configuration for '%s'.", file_name)

    config = fetcher(file_name)

    # Log the keys fetched to avoid sensitive data exposure in logs.
    try:
        keys = ", ".join(sorted(config.keys()))
    except Exception:
        keys = "<uninspectable>"
    logger.debug("Fetched configuration for '%s' with keys: %s", file_name, keys)

    return config


def handle_config_file(file_name: str, sign: str) -> Optional[dict]:
    """
    Read configuration information after validating its cryptographic signature.

    Args:
        file_name: The name of the configuration file to fetch.
        sign: The cryptographic signature provided for the configuration file.

    Returns:
        A dictionary representing the configuration if the signature is verified;
        None if the signature is invalid.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    logger.info("Handling configuration file '%s'.", file_name)

    # Verify signature against trusted signatures (raises IOError if file is unknown).
    if not _verify_signature(file_name, sign, TRUSTED_CONFIG_SIGNATURES):
        logger.debug("Aborting fetch due to invalid signature for '%s'.", file_name)
        return None

    # Signature verified; fetch and return the configuration data.
    return _fetch_configuration(file_name, fetch_config_data)
