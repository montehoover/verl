import logging
from typing import Optional, Dict

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a file handler
handler = logging.FileHandler('iot_device.log')
handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(handler)


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

def is_signature_valid(device_id: str, signature: str, trusted_signatures: Dict[str, str]) -> bool:
    """
    Verifies if the provided signature is valid for the given device ID.

    Args:
        device_id: The identification number of the IoT device.
        signature: The cryptographic signature to verify.
        trusted_signatures: A dictionary mapping device IDs to their trusted signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    trusted_signature = trusted_signatures.get(device_id)
    if trusted_signature is None:
        logger.warning(f"Device ID {device_id} not found in trusted list.")
        return False
    
    is_valid = signature == trusted_signature
    if is_valid:
        logger.info(f"Signature for device {device_id} verified successfully.")
    else:
        logger.warning(f"Signature verification failed for device {device_id}.")
    return is_valid

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
    logger.info(f"Attempting to handle data packet from device {device_id}.")
    if data is None:
        logger.error(f"Data packet from device {device_id} is None. Raising ConnectionError.")
        raise ConnectionError(f"Data packet from device {device_id} not received properly or is missing.")

    if is_signature_valid(device_id, signature, TRUSTED_DEVICE_SIGNATURES):
        logger.info(f"Data packet from device {device_id} processed successfully.")
        return data
    else:
        logger.warning(f"Failed to process data packet from device {device_id} due to invalid signature.")
        return None
