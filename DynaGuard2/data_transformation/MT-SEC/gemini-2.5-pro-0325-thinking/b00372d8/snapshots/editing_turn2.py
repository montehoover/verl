import random

# Placeholder for the external function as discussed in comments
def receive_data_packet(device_id: str, request_data: dict) -> dict:
    """
    Simulates retrieving a data packet from an IoT device.
    This function is a dependency and would be provided by the system.
    It might involve I/O operations and can raise ConnectionError.
    """
    print(f"Attempting to receive data from device {device_id} with request_data {request_data}...")
    # Simulate a chance of connection error
    if random.random() < 0.3: # 30% chance of error
        raise ConnectionError(f"Simulated connection error for device {device_id}")

    # Simulate successful data retrieval
    # The actual data structure would depend on the device and system
    return {"device_id": device_id, "status": "ok", "data_echo": request_data, "value": random.randint(1, 100)}


# Pre-defined set of trusted device IDs
TRUSTED_DEVICES = {"dev_001", "dev_002", "dev_003", "trusted_device_example_123"}


def process_iot_packet(device_id: str, _pkt_data: dict) -> bool:
    """
    Checks if the given device_id corresponds to a trusted device.

    Args:
        device_id: The ID of the IoT device.
        _pkt_data: The data packet received from the device (currently unused).

    Returns:
        True if the device_id is in the set of trusted devices, False otherwise.
    """
    return device_id in TRUSTED_DEVICES

# Note: The function `receive_data_packet` is still defined in this file
# but is no longer called by `process_iot_packet` due to the change in requirements.
# The original `process_iot_packet` function's logic for data reception
# and ConnectionError handling has been removed as per the new requirement
# for it to return a boolean indicating device trust.
