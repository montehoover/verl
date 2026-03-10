import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='iot_processing.log',
    filemode='a'  # Append to the log file
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

def _validate_signature(device_id: str, crypt_sig: str, trusted_signatures: dict) -> bool:
    """
    Validates the cryptographic signature for a given device.

    Args:
        device_id: The ID of the device.
        crypt_sig: The cryptographic signature to validate.
        trusted_signatures: A dictionary of trusted device IDs and their signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected_sig = trusted_signatures.get(device_id)
    # This function assumes device_id might be in trusted_signatures.
    # The check for whether device_id is known at all (and thus can be "received from")
    # is handled by the caller, process_iot_packet, to align with ConnectionError semantics.
    if expected_sig is None:
        # This case should ideally not be hit if process_iot_packet checks first,
        # but as a pure validator, it correctly states signature is not valid for an unknown device.
        return False
    return crypt_sig == expected_sig

def _handle_data_packet(pkt_data: dict) -> dict:
    """
    Processes the data packet.
    (Currently, this function is a placeholder and simply returns the packet data.)

    Args:
        pkt_data: The data packet to process.

    Returns:
        The processed data.
    """
    # In a real scenario, this function would perform actual data processing.
    return pkt_data

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
        ConnectionError: If the packet from the device cannot be received successfully
                         (e.g., device_id is not recognized).
    """
    logger.info(f"Processing packet for device_id: {device_id}. Data: {pkt_data}")

    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        # This check ensures that we only attempt to process packets from known, trusted devices.
        # If the device is not in the trusted list, it's treated as a connection/reception failure.
        logger.warning(f"Device ID {device_id} not in trusted list. Raising ConnectionError.")
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

    logger.info(f"Attempting signature validation for device_id: {device_id}.")
    signature_valid = _validate_signature(device_id, crypt_sig, TRUSTED_DEVICE_SIGNATURES)

    if signature_valid:
        logger.info(f"Signature validation successful for device_id: {device_id}.")
        processed_data = _handle_data_packet(pkt_data)
        logger.info(f"Data packet processed for device_id: {device_id}. Result: {processed_data}")
        return processed_data
    else:
        logger.warning(f"Signature validation failed for device_id: {device_id}.")
        return None      # Signature validation fails
