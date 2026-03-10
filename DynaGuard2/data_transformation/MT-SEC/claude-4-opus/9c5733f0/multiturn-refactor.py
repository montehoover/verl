import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}


def receive_data_packet(id_device: str):
    """
    Simulated function to receive a data packet from the IoT device.
    
    Args:
        id_device (str): The unique identifier for the IoT device.
    
    Returns:
        dict: The data packet received from the device.
    
    Raises:
        ConnectionError: If the device is not in the trusted devices list.
    """
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")


def validate_signature(id_device: str, packet_signature: str) -> bool:
    """
    Validate the cryptographic signature for a given device.
    
    Args:
        id_device (str): The unique identifier for the IoT device.
        packet_signature (str): The cryptographic signature to validate.
    
    Returns:
        bool: True if the signature is valid, False otherwise.
    """
    if id_device not in TRUSTED_DEVICE_SIGNATURES:
        logger.warning(f"Device {id_device} not found in trusted devices")
        return False
    
    is_valid = TRUSTED_DEVICE_SIGNATURES[id_device] == packet_signature
    if is_valid:
        logger.info(f"Signature validation successful for device {id_device}")
    else:
        logger.warning(f"Signature validation failed for device {id_device}")
    
    return is_valid


def get_data_packet(id_device: str) -> dict:
    """
    Retrieve data packet from the IoT device.
    
    Args:
        id_device (str): The unique identifier for the IoT device.
    
    Returns:
        dict: The data packet retrieved from the device.
    
    Raises:
        ConnectionError: If there is an issue receiving data from the device.
    """
    logger.info(f"Attempting to retrieve data packet from device {id_device}")
    try:
        data = receive_data_packet(id_device)
        logger.info(f"Successfully retrieved data packet from device {id_device}")
        return data
    except ConnectionError as e:
        logger.error(f"Failed to retrieve data packet from device {id_device}: {str(e)}")
        raise


def process_packet_data(id_device: str, packet_signature: str, packet_data: dict):
    """
    Authenticate and process data packets from IoT devices.
    
    This function validates the cryptographic signature of a data packet
    before processing it. The device must be in the trusted devices list
    and the signature must match the expected value.
    
    Args:
        id_device (str): The unique identifier for the IoT device sending the data.
        packet_signature (str): The cryptographic signature attached to the data packet.
        packet_data (dict): The actual content of the packet to be processed.
    
    Returns:
        dict or None: The data as a dictionary if the cryptographic signature is valid;
                     otherwise, returns None.
    
    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device.
    """
    logger.info(f"Processing packet from device {id_device}")
    
    # Verify device is trusted (will raise ConnectionError if not)
    try:
        get_data_packet(id_device)
    except ConnectionError:
        logger.error(f"Connection error for device {id_device}")
        raise ConnectionError(f"Failed to receive packet from device {id_device}")
    
    # Validate the cryptographic signature
    if validate_signature(id_device, packet_signature):
        # Signature is valid, return the packet data
        logger.info(f"Packet processed successfully for device {id_device}")
        return packet_data
    else:
        # Invalid signature, return None
        logger.warning(f"Packet rejected due to invalid signature for device {id_device}")
        return None
