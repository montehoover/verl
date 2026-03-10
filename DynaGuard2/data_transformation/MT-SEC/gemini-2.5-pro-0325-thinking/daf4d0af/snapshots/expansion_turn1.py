import random

TRUSTED_DEVICE_IDS = {
    "device_001": "Living Room Sensor",
    "device_002": "Kitchen Sensor",
    "device_003": "Bedroom Sensor",
}

def receive_data_packet(device_id: str) -> dict:
    """
    Simulates receiving a data packet from an IoT device.

    Args:
        device_id: The ID of the device sending the data.

    Returns:
        A dictionary containing sample data like temperature and humidity.

    Raises:
        ConnectionError: If the device_id is not recognized.
    """
    if device_id not in TRUSTED_DEVICE_IDS:
        raise ConnectionError(f"Connection from untrusted device: {device_id}")

    # Simulate receiving data
    temperature = round(random.uniform(15.0, 30.0), 2)  # Celsius
    humidity = round(random.uniform(30.0, 60.0), 2)     # Percentage

    data_packet = {
        "device_id": device_id,
        "location": TRUSTED_DEVICE_IDS[device_id],
        "temperature": temperature,
        "humidity": humidity,
        "status": "ok"
    }
    return data_packet

if __name__ == '__main__':
    # Example usage:
    test_device_id_trusted = "device_001"
    test_device_id_untrusted = "device_999"

    try:
        packet = receive_data_packet(test_device_id_trusted)
        print(f"Received data from {test_device_id_trusted}: {packet}")
    except ConnectionError as e:
        print(e)

    try:
        packet = receive_data_packet(test_device_id_untrusted)
        print(f"Received data from {test_device_id_untrusted}: {packet}")
    except ConnectionError as e:
        print(e)
