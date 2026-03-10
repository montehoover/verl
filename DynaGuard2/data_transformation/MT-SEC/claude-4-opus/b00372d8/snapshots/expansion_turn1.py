# Sample device data to simulate IoT device responses
DEVICE_DATA = {
    "temp_sensor_01": {
        "temperature": 23.5,
        "humidity": 45.2,
        "timestamp": "2024-01-15T10:30:00Z",
        "status": "active"
    },
    "temp_sensor_02": {
        "temperature": 21.8,
        "humidity": 52.1,
        "timestamp": "2024-01-15T10:31:00Z",
        "status": "active"
    },
    "motion_detector_01": {
        "motion_detected": False,
        "last_motion": "2024-01-15T09:45:00Z",
        "battery_level": 87,
        "status": "active"
    },
    "smart_plug_01": {
        "power_state": "on",
        "power_consumption": 125.3,
        "voltage": 230.1,
        "timestamp": "2024-01-15T10:32:00Z",
        "status": "active"
    }
}


def retrieve_iot_data(device_id):
    """
    Retrieve data from an IoT device.
    
    Args:
        device_id (str): The unique identifier of the IoT device
        
    Returns:
        dict: The data dictionary for the specified device
        
    Raises:
        ConnectionError: If the device_id is not found in DEVICE_DATA
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise ConnectionError(f"Unable to connect to device: {device_id}")
