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

def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict):
    """
    Validates and processes data packets from IoT devices.

    Each packet is accompanied by a cryptographic signature that must be
    authenticated before handling the data.

    Args:
        device_id: A unique identifier for the IoT device sending the data.
        crypt_sig: The cryptographic signature sent along with the packet.
        pkt_data: The content of the data packet that was signed. (Note:
                  this implementation returns data from receive_data_packet,
                  not this pkt_data, per example).

    Returns:
        The processed data (from receive_data_packet) as a dictionary if the
        signature validation passes. Otherwise, it returns None.

    Raises:
        ConnectionError: If the packet from the device cannot be received
                         successfully by receive_data_packet.
    """
    expected_sig = TRUSTED_DEVICE_SIGNATURES.get(device_id)

    if expected_sig is None or expected_sig != crypt_sig:
        # Device not trusted or signature mismatch
        return None
    else:
        # Signature is valid, attempt to receive/process the data
        # receive_data_packet will raise ConnectionError if it fails
        processed_data = receive_data_packet(device_id)
        return processed_data
