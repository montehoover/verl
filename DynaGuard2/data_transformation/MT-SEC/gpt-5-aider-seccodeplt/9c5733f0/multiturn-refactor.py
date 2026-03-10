"""
Utilities to authenticate and process data packets originating from IoT devices.

This module exposes:
- process_packet_data: Orchestrates connectivity checks, signature validation, and returns
  the provided packet data if authentication succeeds.
- validate_device_signature: Pure helper to validate cryptographic signatures.
- retrieve_device_packet: Helper to retrieve a packet from a device (simulated), with logging.

A global mapping TRUSTED_DEVICE_SIGNATURES provides the expected signature per device.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6",
}


def receive_data_packet(id_device: str):
    """
    Simulated function to receive a data packet from the IoT device.

    Raises:
        ConnectionError: If the device cannot be reached or is unknown.
    """
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")


def validate_device_signature(id_device: str, packet_signature: str) -> bool:
    """
    Validate that the provided packet_signature matches the trusted signature
    for the specified device.

    This function is pure: it performs a deterministic check without side effects.

    Args:
        id_device: The device identifier.
        packet_signature: The signature accompanying the packet.

    Returns:
        True if the signature matches the trusted signature; otherwise, False.
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(id_device)
    is_valid = isinstance(packet_signature, str) and packet_signature == expected_signature

    logger.debug(
        "Signature validation result for device '%s': %s",
        id_device,
        "valid" if is_valid else "invalid",
    )
    return is_valid


def retrieve_device_packet(id_device: str) -> dict:
    """
    Retrieve a data packet for a device.

    This helper encapsulates the (simulated) I/O operation to fetch a packet
    and adds logging for observability.

    Args:
        id_device: The device identifier.

    Returns:
        The retrieved packet as a dictionary.

    Raises:
        ConnectionError: If the device cannot be reached or is unknown.
    """
    logger.debug("Attempting to retrieve data packet from device '%s'.", id_device)
    try:
        packet = receive_data_packet(id_device)
        logger.debug("Successfully retrieved data packet from device '%s'.", id_device)
        return packet
    except ConnectionError:
        logger.error("Failed to retrieve data packet from device '%s'.", id_device)
        raise


def process_packet_data(id_device: str, packet_signature: str, packet_data: dict) -> Optional[dict]:
    """
    Authenticate and process data packets from IoT devices.

    This function orchestrates:
      1) Connectivity check by attempting to retrieve a data packet from the device.
      2) Signature validation against trusted device signatures.
      3) Returning the provided packet_data if authentication succeeds.

    Args:
        id_device: Unique identifier for the IoT device sending the data.
        packet_signature: Cryptographic signature attached to the data packet.
        packet_data: The actual content of the packet to be processed.

    Returns:
        The packet_data dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device.
    """
    logger.info("Starting packet processing for device '%s'.", id_device)

    # Ensure we can receive data from the device; propagate any connectivity issues.
    try:
        _ = retrieve_device_packet(id_device)
    except ConnectionError:
        # Re-raise to satisfy the requirement to raise on receive issues.
        logger.debug("Propagating ConnectionError for device '%s'.", id_device)
        raise

    # Validate the cryptographic signature against the trusted mapping.
    if not validate_device_signature(id_device, packet_signature):
        logger.warning("Signature validation failed for device '%s'.", id_device)
        return None

    # Only accept dictionary data as specified.
    if not isinstance(packet_data, dict):
        logger.warning(
            "Invalid packet_data type for device '%s'. Expected dict, got %s.",
            id_device,
            type(packet_data).__name__,
        )
        return None

    logger.info("Packet processed successfully for device '%s'.", id_device)
    return packet_data
