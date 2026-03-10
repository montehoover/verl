from typing import Any, Dict, Optional

# Placeholder globals to satisfy static analysis; actual values should be provided externally.
TRUSTED_DEVICE_SIGNATURES: Dict[str, str] = {}


def receive_data_packet(id_device: str):
    # Placeholder function; actual implementation should be provided externally.
    raise ConnectionError(f"Failed to receive packet from device {id_device}")


def process_packet_data(id_device: str, packet_signature: str, packet_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Authenticate and process IoT device packet data.

    Validates that the provided packet_signature matches the trusted signature
    for the given id_device in the global TRUSTED_DEVICE_SIGNATURES mapping.
    If the signature is invalid, returns None.

    If the signature is valid, attempts to receive a data packet from the device
    to ensure connectivity. If receiving the data packet fails, a ConnectionError
    is raised. On success, returns the provided packet_data dictionary.
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(id_device)
    if expected_signature is None or expected_signature != packet_signature:
        return None

    if not isinstance(packet_data, dict):
        return None

    # Attempt to receive data from the device to detect connection issues.
    _ = receive_data_packet(id_device)  # May raise ConnectionError

    return packet_data
