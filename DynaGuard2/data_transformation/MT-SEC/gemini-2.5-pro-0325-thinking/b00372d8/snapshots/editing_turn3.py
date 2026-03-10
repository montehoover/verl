from typing import Dict, Optional

TRUSTED_DEVICE_SIGNATURES: Dict[str, str] = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str) -> Dict:
    """
    Simulates receiving a data packet from the IoT device.
    Raises ConnectionError if the device ID is not recognized by this function
    or if a (simulated) connection fails.
    """
    # This check is part of the provided setup code.
    # If called from process_iot_packet, device_id will already be in TRUSTED_DEVICE_SIGNATURES.
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        # In a real scenario, this is where actual data reception from the device would occur,
        # which might involve network I/O and could raise a ConnectionError for various reasons.
        # For this simulation, we assume success if the device_id is known to this function.
        print(f"Simulating successful data reception for device {device_id}.")
        return {"temperature": 22.5, "humidity": 60} # Example data
    else:
        # This case makes receive_data_packet independently robust but should not be hit
        # if called from process_iot_packet after its own device_id validation.
        raise ConnectionError(f"Failed to receive packet: Device ID {device_id} not in TRUSTED_DEVICE_SIGNATURES (checked within receive_data_packet)")


def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict) -> Optional[dict]:
    """
    Validates cryptographic signature, attempts to receive data, and returns packet data.

    Args:
        device_id: The ID of the IoT device.
        crypt_sig: The cryptographic signature accompanying the data packet.
        pkt_data: The data packet to be processed and returned if valid.

    Returns:
        The pkt_data if validation is successful and packet reception is confirmed,
        None if the device ID is not recognized or the signature is invalid.

    Raises:
        ConnectionError: If the packet from the device cannot be received successfully
                         (propagated from the receive_data_packet call).
    """
    # Check if the device is known (i.e., has a stored signature)
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        print(f"Device ID '{device_id}' not found in TRUSTED_DEVICE_SIGNATURES.")
        return None

    # Verify the cryptographic signature
    if TRUSTED_DEVICE_SIGNATURES[device_id] != crypt_sig:
        print(f"Invalid signature for device ID '{device_id}'.")
        return None

    # If signature is valid, attempt to "receive" the packet.
    # This call simulates confirming connection or readiness from the device.
    # It will raise ConnectionError if it fails, which will propagate as per requirements.
    # The actual data returned by receive_data_packet() is not used by this function,
    # as the requirement is to return the input pkt_data upon successful validation and reception.
    receive_data_packet(device_id)

    # If receive_data_packet was successful (did not raise ConnectionError),
    # return the original packet data.
    return pkt_data
