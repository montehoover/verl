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
        device_id: str, a unique identifier for the IoT device sending the data.
        crypt_sig: str, the cryptographic signature sent along with the packet.
        pkt_data: dict, the content of the data packet to be processed.

    Returns:
        The processed data as a dictionary if the signature validation passes.
        Otherwise, it returns None.

    Raises:
        ConnectionError: If the packet from the device cannot be received successfully.
    """
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        # If the device_id is not in the trusted list, it implies that a packet
        # from such a device cannot be successfully received or processed.
        # This aligns with the behavior of receive_data_packet.
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

    expected_sig = TRUSTED_DEVICE_SIGNATURES[device_id]
    if crypt_sig == expected_sig:
        return pkt_data  # Signature validation passes
    else:
        return None      # Signature validation fails
