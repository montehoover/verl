import random
from datetime import datetime

# Sample dictionary of trusted device IDs
TRUSTED_DEVICES = {
    "IOT001": "Temperature Sensor - Living Room",
    "IOT002": "Humidity Sensor - Bathroom",
    "IOT003": "Smart Thermostat - Main Floor",
    "IOT004": "Weather Station - Outdoor",
    "IOT005": "Air Quality Monitor - Bedroom"
}

def receive_data_packet(device_id):
    """
    Simulates receiving a data packet from an IoT device.
    
    Args:
        device_id (str): The unique identifier of the IoT device
        
    Returns:
        dict: A dictionary containing sample sensor data
        
    Raises:
        ConnectionError: If the device_id is not recognized
    """
    if device_id not in TRUSTED_DEVICES:
        raise ConnectionError(f"Device {device_id} is not a trusted device")
    
    # Generate sample data based on device type
    data_packet = {
        "device_id": device_id,
        "device_name": TRUSTED_DEVICES[device_id],
        "timestamp": datetime.now().isoformat(),
        "temperature": round(random.uniform(15.0, 30.0), 2),
        "humidity": round(random.uniform(30.0, 70.0), 2),
        "battery_level": random.randint(20, 100),
        "signal_strength": random.randint(-90, -30)
    }
    
    # Add device-specific data
    if "Temperature" in TRUSTED_DEVICES[device_id]:
        data_packet["unit"] = "celsius"
    elif "Humidity" in TRUSTED_DEVICES[device_id]:
        data_packet["unit"] = "percentage"
    elif "Thermostat" in TRUSTED_DEVICES[device_id]:
        data_packet["target_temperature"] = round(random.uniform(18.0, 25.0), 1)
        data_packet["mode"] = random.choice(["heating", "cooling", "auto"])
    elif "Weather" in TRUSTED_DEVICES[device_id]:
        data_packet["pressure"] = round(random.uniform(980.0, 1040.0), 1)
        data_packet["wind_speed"] = round(random.uniform(0.0, 25.0), 1)
    elif "Air Quality" in TRUSTED_DEVICES[device_id]:
        data_packet["pm25"] = random.randint(0, 150)
        data_packet["co2"] = random.randint(400, 2000)
    
    return data_packet

# Example usage
if __name__ == "__main__":
    # Test with a valid device
    try:
        data = receive_data_packet("IOT001")
        print("Received data:", data)
    except ConnectionError as e:
        print("Error:", e)
    
    # Test with an invalid device
    try:
        data = receive_data_packet("UNKNOWN")
        print("Received data:", data)
    except ConnectionError as e:
        print("Error:", e)
