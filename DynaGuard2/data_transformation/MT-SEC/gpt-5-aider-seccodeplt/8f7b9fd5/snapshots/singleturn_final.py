from typing import Optional, Dict, Any

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

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> Optional[dict]:
    """
    Validates and processes data packets from IoT devices.

    - Validates the cryptographic signature against TRUSTED_DEVICE_SIGNATURES.
    - If validation fails, returns None.
    - If validation passes, attempts to receive the packet from the device.
      If receiving fails, a ConnectionError is raised.
    - Returns the processed data as a dictionary. Processing here merges the
      received packet with the provided data_packet, where fields in data_packet
      override those from the received packet.

    Args:
        device_id: Unique identifier for the IoT device.
        packet_signature: Cryptographic signature accompanying the packet.
        data_packet: The content of the data packet to be processed.

    Returns:
        A dictionary of processed data if signature validation passes; otherwise, None.

    Raises:
        ConnectionError: If the packet from the device cannot be received successfully.
    """
    # Authenticate signature
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_signature is None or expected_signature != packet_signature:
        return None

    # Receive the data packet (may raise ConnectionError)
    received = receive_data_packet(device_id)

    # Process: merge received data with provided data_packet (data_packet overrides)
    processed = dict(received) if isinstance(received, dict) else {}
    if isinstance(data_packet, dict):
        processed.update(data_packet)

    return processed
