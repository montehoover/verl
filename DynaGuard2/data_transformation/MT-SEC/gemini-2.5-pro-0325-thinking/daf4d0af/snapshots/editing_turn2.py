def process_iot_data(device_id: str, data: dict, approved_devices: list[str]) -> bool:
    """
    Processes IoT data by checking if the device ID is approved.

    Args:
        device_id: The ID of the IoT device.
        data: A dictionary containing the data from the IoT device. (Unused in current logic)
        approved_devices: A list of approved device IDs.

    Returns:
        True if the device_id is in approved_devices, False otherwise.
    """
    return device_id in approved_devices
