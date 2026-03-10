import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}


# Simulated function to receive a data packet from the IoT device
def receive_data_packet(id_device: str) -> dict:
    """
    Simulates receiving a data packet from an IoT device.

    This function checks if the device ID is known. If so, it returns
    simulated sensor data. Otherwise, it raises a ConnectionError.

    Args:
        id_device: The unique identifier for the IoT device.

    Returns:
        A dictionary containing simulated sensor data if the device is known.

    Raises:
        ConnectionError: If the id_device is not found in TRUSTED_DEVICE_SIGNATURES,
                         simulating a failure to receive a packet.
    """
    logging.info(f"Attempting to receive data packet from device: {id_device}")
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        data = {"temperature": 22.5, "humidity": 60}
        logging.info(f"Successfully received data packet from device: {id_device}")
        return data
    else:
        logging.error(f"Failed to receive packet from unknown device: {id_device}")
        raise ConnectionError(f"Failed to receive packet from device {id_device}")


def _is_signature_valid(id_device: str, packet_signature: str, trusted_signatures: dict) -> bool:
    """
    Validates if the provided packet signature matches the trusted signature for the device.

    Args:
        id_device: The unique identifier for the IoT device.
        packet_signature: The cryptographic signature attached to the data packet.
        trusted_signatures: A dictionary where keys are device IDs and values are
                            their corresponding trusted cryptographic signatures.

    Returns:
        bool: True if the `packet_signature` matches the entry for `id_device`
              in `trusted_signatures`, False otherwise. Returns False if `id_device`
              is not found in `trusted_signatures`.
    """
    logging.info(f"Validating signature for device: {id_device}")
    if id_device not in trusted_signatures:
        logging.warning(f"Device ID {id_device} not found in trusted signatures.")
        return False

    is_valid = packet_signature == trusted_signatures[id_device]
    if is_valid:
        logging.info(f"Signature for device {id_device} is valid.")
    else:
        logging.warning(f"Signature mismatch for device {id_device}.")
    return is_valid


def process_packet_data(id_device: str, packet_signature: str, packet_data: dict) -> dict | None:
    """
    Authenticates and processes data packets originating from IoT devices.

    This function first attempts to simulate data reception (which can raise
    ConnectionError). If successful, it validates the cryptographic signature
    of the packet. If the signature is valid, the packet data is returned.
    Otherwise, None is returned.

    Args:
        id_device: The unique identifier for the IoT device sending the data.
        packet_signature: The cryptographic signature attached to the data packet.
        packet_data: The actual content of the packet to be processed.

    Returns:
        dict | None: The `packet_data` as a dictionary if the cryptographic
                     signature is valid; otherwise, returns None.

    Raises:
        ConnectionError: If `receive_data_packet` encounters an issue receiving
                         data from the IoT device.
    """
    logging.info(f"Processing packet data for device: {id_device}")
    try:
        # Simulate data reception attempt as per requirements.
        # The actual data from this call isn't used for processing,
        # but the call itself is a prerequisite.
        receive_data_packet(id_device)
        logging.info(f"Data reception check successful for device: {id_device}")

    except ConnectionError as e:
        logging.error(f"ConnectionError during data reception check for device {id_device}: {e}")
        # Propagate the ConnectionError if receive_data_packet raises it.
        raise

    # If data reception check (simulated) succeeded, proceed with signature validation.
    if _is_signature_valid(id_device, packet_signature, TRUSTED_DEVICE_SIGNATURES):
        logging.info(f"Packet data processed successfully for device: {id_device}")
        return packet_data
    else:
        logging.warning(f"Invalid signature. Packet data processing failed for device: {id_device}")
        return None
