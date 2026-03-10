from typing import Any, Dict


def process_packet_data(id_device: str, packet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch and return the packet data for a given IoT device.

    This function looks up the packet data in the provided dictionary using the
    device ID and returns it as a dictionary. It handles any failures gracefully
    by returning an empty dictionary.

    Parameters:
        id_device: The device identifier as a string.
        packet_data: A dictionary mapping device IDs to their packet data.

    Returns:
        A dictionary containing the packet data for the given device ID.
        Returns an empty dictionary if the data is unavailable or on failure.
    """
    try:
        # Ensure we have a key-compatible string
        key = id_device if isinstance(id_device, str) else str(id_device)

        # Validate input structure
        if not isinstance(packet_data, dict):
            return {}

        data = packet_data.get(key, {})

        # Return only dictionary payloads; coerce failures to empty dict
        return data if isinstance(data, dict) else {}
    except Exception:
        # Handle any unexpected issues (e.g., simulated connection failures) gracefully
        return {}
