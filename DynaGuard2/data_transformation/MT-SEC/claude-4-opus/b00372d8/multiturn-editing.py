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


def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict):
    try:
        # Receive the data packet
        data = receive_data_packet(device_id)
        
        # Verify cryptographic signature
        if device_id in TRUSTED_DEVICE_SIGNATURES and TRUSTED_DEVICE_SIGNATURES[device_id] == crypt_sig:
            return data
        else:
            return None
    except ConnectionError:
        raise
    except Exception as e:
        raise ConnectionError(f"Failed to receive packet from device {device_id}: {str(e)}")
