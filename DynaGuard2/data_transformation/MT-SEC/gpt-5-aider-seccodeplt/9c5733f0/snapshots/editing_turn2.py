from typing import Any, Dict

# Global dictionary holding trusted device status. Should be populated elsewhere.
TRUSTED_DEVICES: Dict[str, bool] = {}


def process_packet_data(id_device: str, packet_data: Dict[str, Any]) -> bool:
    """
    Verify the device is trusted before processing packet data.

    Checks the given device ID against a global trusted devices dictionary.
    If the device is not trusted, returns False. If trusted, attempts to
    access the packet data and returns True on success. Any issues (e.g.,
    invalid inputs or unexpected errors) are handled gracefully by returning False.

    Parameters:
        id_device: The device identifier as a string.
        packet_data: A dictionary mapping device IDs to their packet data.

    Returns:
        False if the device is not trusted or on failure; True otherwise.
    """
    try:
        key = id_device if isinstance(id_device, str) else str(id_device)

        # Check trust list
        if not TRUSTED_DEVICES.get(key, False):
            return False

        # Validate input structure and "process" packet data access
        if not isinstance(packet_data, dict):
            return False

        _ = packet_data.get(key, {})

        return True
    except Exception:
        return False
