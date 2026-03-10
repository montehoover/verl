import random
from datetime import datetime
import hashlib
import json

# Sample dictionary of trusted device IDs
TRUSTED_DEVICES = {
    "IOT001": "Temperature Sensor - Living Room",
    "IOT002": "Humidity Sensor - Bathroom",
    "IOT003": "Smart Thermostat - Main Floor",
    "IOT004": "Weather Station - Outdoor",
    "IOT005": "Air Quality Monitor - Bedroom"
}

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
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

def verify_packet_signature(data, signature):
    """
    Verifies the integrity of a data packet by comparing its hash with a signature.
    
    Args:
        data (str): The data to verify
        signature (str): The expected signature (hash) of the data
        
    Returns:
        bool: True if the signature matches, False otherwise
    """
    try:
        # Compute SHA-256 hash of the data
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Compare the computed hash with the provided signature
        return data_hash == signature
    except Exception:
        # Handle any unsupported operations
        return False

def handle_data_packet(device_id, signature, data):
    """
    Processes data packets received from an IoT device.
    
    Args:
        device_id (str): The unique identifier of the IoT device
        signature (str): The cryptographic signature of the data
        data (dict): The data packet received from the device
        
    Returns:
        dict or None: The data as a dictionary if the signature is valid, otherwise None
    """
    # Convert data dictionary to string for signature verification
    data_str = json.dumps(data, sort_keys=True)
    
    # Verify the signature
    if verify_packet_signature(data_str, signature):
        return data
    else:
        return None

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
