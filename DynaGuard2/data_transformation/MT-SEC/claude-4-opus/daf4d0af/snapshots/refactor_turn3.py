import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iot_data_packets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def verify_signature(device_id: str, signature: str) -> bool:
    """
    Verify if the provided signature matches the trusted signature for the device.
    
    Args:
        device_id: The identification number of the IoT device
        signature: The cryptographic signature to verify
        
    Returns:
        True if the signature is valid, False otherwise
    """
    return device_id in TRUSTED_DEVICE_SIGNATURES and TRUSTED_DEVICE_SIGNATURES[device_id] == signature

def handle_data_packet(device_id: str, signature: str, data: dict):
    try:
        logger.info(f"Attempting to receive data packet from device {device_id}")
        
        # Attempt to receive the data packet
        received_data = receive_data_packet(device_id)
        logger.info(f"Successfully received data packet from device {device_id}")
        
        # Verify the signature
        if verify_signature(device_id, signature):
            logger.info(f"Signature verification successful for device {device_id}")
            # Signature is valid, return the data
            return data
        else:
            logger.warning(f"Signature verification failed for device {device_id}")
            # Invalid signature
            return None
            
    except ConnectionError as e:
        logger.error(f"Connection error while receiving packet from device {device_id}: {str(e)}")
        # Re-raise the ConnectionError if packet reception fails
        raise
