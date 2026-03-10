from typing import Optional, Dict

TRUSTED_DEVICE_SIGNATURES: Dict[str, str] = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str) -> Dict:
    """
    Simulates receiving a data packet from an IoT device.
    Raises ConnectionError if the device_id is not recognized (simulating reception failure).
    """
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        # In a real scenario, this data would come from the device.
        # The signature would also be part of the incoming packet or headers.
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def handle_data_packet(device_id: str, signature: str, data: Optional[Dict]) -> Optional[Dict]:
    """
    Processes data packets received from an IoT device.
    Verifies the cryptographic signature before trusting the data.

    Args:
        device_id: The identification number of the IoT device.
        signature: The cryptographic signature accompanying the packet.
        data: The payload of the packet. If None, indicates packet not received properly.

    Returns:
        The data as a dictionary if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If the data packet is indicated as not received properly (i.e., data is None).
    """
    if data is None:
        raise ConnectionError(f"Data packet from device {device_id} not received properly or is missing.")

    trusted_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)

    if trusted_signature is None:
        # Device ID not found in the trusted list.
        # This implies the signature cannot be verified against a known trusted one.
        return None

    if signature == trusted_signature:
        # Signature matches the trusted signature for this device.
        return data
    else:
        # Signature does not match.
        return None
