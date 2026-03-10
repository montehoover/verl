from typing import Any, Dict, Optional

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")


def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validates and processes a data packet from an IoT device.

    Args:
        device_id: Unique identifier for the IoT device sending the data.
        crypt_sig: Cryptographic signature accompanying the packet.
        pkt_data: The content of the data packet to be processed.

    Returns:
        The processed data as a dictionary if the signature validation passes.
        Otherwise, returns None.

    Raises:
        ConnectionError: If the packet from the device cannot be received successfully.
    """
    # Validate the cryptographic signature for the device
    expected_sig = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_sig is None or crypt_sig != expected_sig:
        return None

    # Attempt to receive the packet; propagate ConnectionError on failure
    received_data = receive_data_packet(device_id)

    # Process/merge data: prefer device-received values as authoritative
    # Shallow merge: pkt_data keys are included but overridden by received_data on conflicts.
    if not isinstance(pkt_data, dict):
        pkt_data = {}

    processed: Dict[str, Any] = dict(pkt_data)
    processed.update(received_data)

    return processed
