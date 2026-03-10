from typing import Optional

def process_packet_data(id_device: str, packet_signature: str, packet_data: dict) -> Optional[dict]:
    """
    Authenticate and process data packets originating from IoT devices.

    Behavior:
    - Validate the provided packet_signature against TRUSTED_DEVICE_SIGNATURES for id_device.
    - If the signature is valid, fetch and return the actual packet data via receive_data_packet.
    - If the signature is invalid or the device is untrusted/unknown, return None.
    - If there is an issue receiving data from the IoT device, raise ConnectionError.

    Note:
    - Expects the following to exist in the module's global scope:
        - TRUSTED_DEVICE_SIGNATURES: Dict[str, str]
        - receive_data_packet(id_device: str) -> dict
    """
    # Fetch trusted signatures from globals without referencing an undefined name directly.
    trusted_signatures = globals().get("TRUSTED_DEVICE_SIGNATURES", {})
    if not isinstance(trusted_signatures, dict):
        # If the global is not a dict, treat as no trusted devices configured.
        return None

    # Validate expected signature for the device
    expected_signature = trusted_signatures.get(id_device)
    if expected_signature is None:
        # Unknown/untrusted device
        return None

    if packet_signature != expected_signature:
        # Signature mismatch; do not trust the packet
        return None

    # Signature is valid; attempt to receive and return the packet data.
    receive_fn = globals().get("receive_data_packet")
    if not callable(receive_fn):
        # Treat absence of the receiving function as an issue receiving data.
        raise ConnectionError("Receiving function not available")

    # Let ConnectionError from receive_fn propagate as specified.
    data = receive_fn(id_device)

    # Ensure we return a dictionary as specified.
    return data if isinstance(data, dict) else None
