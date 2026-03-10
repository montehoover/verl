import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def validate_signature(device_id: str, packet_signature: str) -> bool:
    """Validate the cryptographic signature for a given device."""
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        logger.warning(f"Unknown device ID: {device_id}")
        return False
    
    is_valid = TRUSTED_DEVICE_SIGNATURES[device_id] == packet_signature
    if is_valid:
        logger.info(f"Signature validation successful for device {device_id}")
    else:
        logger.warning(f"Signature validation failed for device {device_id}")
    
    return is_valid

def process_data(device_id: str, data_packet: dict) -> dict:
    """Process the data packet and return formatted result."""
    result = {
        "device_id": device_id,
        "status": "processed",
        "data": data_packet
    }
    logger.info(f"Data packet processed successfully for device {device_id}")
    return result

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> dict:
    logger.info(f"Processing packet from device {device_id}")
    
    # Validate the signature
    if not validate_signature(device_id, packet_signature):
        logger.error(f"Packet processing failed for device {device_id}: Invalid signature")
        return None
    
    # Receive the data packet
    try:
        received_packet = receive_data_packet(device_id)
        logger.info(f"Data packet received successfully from device {device_id}")
    except ConnectionError as e:
        logger.error(f"Connection error for device {device_id}: {str(e)}")
        raise
    
    # Process the data packet
    result = process_data(device_id, data_packet)
    logger.info(f"Packet processing completed for device {device_id}")
    return result
