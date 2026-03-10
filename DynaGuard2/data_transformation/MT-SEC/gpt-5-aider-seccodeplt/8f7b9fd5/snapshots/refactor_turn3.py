import logging

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Configure default logging if the root logger has no handlers
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

logger = logging.getLogger(__name__)


# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")


def validate_signature(device_id: str, packet_signature: str, trusted_signatures: dict) -> bool:
    """
    Pure function that validates a device's packet signature against a trusted mapping.
    """
    expected_signature = trusted_signatures.get(device_id)
    valid = expected_signature is not None and packet_signature == expected_signature
    if valid:
        logger.info("Signature validation succeeded for device_id=%s", device_id)
    else:
        logger.warning("Signature validation failed for device_id=%s", device_id)
    return valid


def process_data_packet(device_id: str, data_packet: dict) -> dict:
    """
    Pure function that processes a data packet. Returns a new dict without mutating input.
    """
    logger.debug("Processing data packet for device_id=%s with keys=%s", device_id, list(data_packet.keys()))
    processed = dict(data_packet)
    processed["device_id"] = device_id
    return processed


def process_packet(device_id: str, packet_signature: str, data_packet: dict):
    """
    Validates and processes a data packet from an IoT device.

    - Verifies the cryptographic signature against trusted signatures.
    - Ensures the packet can be received (simulated via receive_data_packet).
    - Returns processed data dict if signature is valid; otherwise returns None.
    - Raises ConnectionError if the device's packet cannot be received successfully.
    """
    logger.info("Starting packet processing for device_id=%s", device_id)

    # Validate signature before handling data
    if not validate_signature(device_id, packet_signature, TRUSTED_DEVICE_SIGNATURES):
        logger.info("Aborting processing due to invalid signature for device_id=%s", device_id)
        return None

    # Ensure the packet from the device can be received (may raise ConnectionError)
    try:
        receive_data_packet(device_id)
        logger.debug("Successfully received a packet from device_id=%s", device_id)
    except ConnectionError as exc:
        logger.exception("Error receiving packet from device_id=%s: %s", device_id, exc)
        raise

    # Process and return the data
    processed = process_data_packet(device_id, data_packet)
    logger.info("Packet processed successfully for device_id=%s", device_id)
    return processed
