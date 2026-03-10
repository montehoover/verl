import logging
import os

# Configure logging to a file in the current working directory
logger = logging.getLogger("iot_device")
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_path = os.path.join(os.getcwd(), "iot_packets.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    logger.info("Attempting to receive data packet from device_id=%s", device_id)
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        payload = {"temperature": 22.5, "humidity": 60}
        logger.info("Successfully received data packet from device_id=%s: %s", device_id, payload)
        return payload
    else:
        logger.error("Failed to receive packet from device_id=%s (untrusted or unknown device)", device_id)
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def verify_signature(device_id: str, signature: str, trusted_signatures: dict = TRUSTED_DEVICE_SIGNATURES) -> bool:
    """
    Pure function to verify that the provided signature matches the trusted signature
    for the given device_id.

    Args:
        device_id: Identification number of the IoT device.
        signature: Cryptographic signature accompanying the packet.
        trusted_signatures: Mapping of device IDs to their trusted signatures.

    Returns:
        True if the signature matches the trusted signature for the device; otherwise, False.
    """
    expected = trusted_signatures.get(device_id)
    return expected is not None and signature == expected

def handle_data_packet(device_id: str, signature: str, data: dict):
    """
    Process a data packet received from an IoT device.

    Args:
        device_id: Identification number of the IoT device.
        signature: Cryptographic signature accompanying the packet.
        data: Payload of the packet to be processed.

    Returns:
        The data dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the data packet is not received properly.
    """
    logger.info("handle_data_packet invoked for device_id=%s", device_id)

    try:
        # Attempt to receive a packet to confirm proper reception/connectivity.
        # This will raise ConnectionError if the device is not reachable/valid.
        received_payload = receive_data_packet(device_id)
        logger.info("Connectivity confirmed for device_id=%s; sample payload=%s", device_id, received_payload)
    except ConnectionError:
        logger.exception("Connection error while receiving data packet from device_id=%s", device_id)
        raise

    # Verify the signature using the pure function.
    is_valid = verify_signature(device_id, signature)
    if is_valid:
        logger.info("Signature verification succeeded for device_id=%s", device_id)
        return data
    else:
        logger.warning("Signature verification failed for device_id=%s", device_id)
        return None
