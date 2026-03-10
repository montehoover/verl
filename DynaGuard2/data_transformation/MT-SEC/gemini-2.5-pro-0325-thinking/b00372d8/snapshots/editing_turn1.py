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


def process_iot_packet(device_id: str, pkt_data: dict):
    """
    Receives and processes data packets from IoT devices.

    Args:
        device_id: The ID of the IoT device.
        pkt_data: The data packet received from the device.

    Returns:
        The received data packet.

    Raises:
        ConnectionError: If the packet cannot be received due to a connection issue.
    """
    try:
        # Assuming receive_data_packet is a pre-existing function
        # that handles the actual data reception and might raise ConnectionError.
        # For the purpose of this function, we'll simulate its behavior
        # or call it if it were defined elsewhere.
        # If receive_data_packet is meant to be called with pkt_data,
        # it would look like: received_data = receive_data_packet(device_id, pkt_data)
        # However, the prompt says "retrieves data from devices", implying it might not need pkt_data as input
        # but rather uses device_id to fetch new data.
        # Let's assume receive_data_packet(device_id) is the intended call.
        # And pkt_data is perhaps metadata about the *request* to receive, not the data *itself*.
        # The prompt says "return the received data as is".
        # If pkt_data is the data *to be processed* after being received,
        # then receive_data_packet might be a preliminary step.

        # Clarification: "The function should take a device_id as a string and pkt_data as a dictionary.
        # It should return the received data as is."
        # This implies pkt_data is the data that has been received.
        # "The system can rely on a function called receive_data_packet, which retrieves data from devices."
        # This is slightly ambiguous. Let's assume receive_data_packet is called *within* this function
        # to get the data, and pkt_data is perhaps some configuration or initial payload.
        # Or, pkt_data *is* the data, and receive_data_packet is a function that *was* called to get it.

        # Re-interpreting: "process_iot_packet ... receives and processes data packets"
        # "take ... pkt_data as a dictionary" -> pkt_data is an input to process_iot_packet.
        # "return the received data as is" -> return pkt_data.
        # "If the packet cannot be received due to a connection issue, the function should raise a ConnectionError."
        # "The system can rely on a function called receive_data_packet, which retrieves data from devices."

        # Option 1: pkt_data is the data already received, and receive_data_packet is a helper that was used to get it.
        # In this case, process_iot_packet might just validate or log it.
        # But the prompt says "return the received data as is", which would be pkt_data.
        # The ConnectionError would have to be raised by receive_data_packet.

        # Option 2: process_iot_packet calls receive_data_packet.
        # def receive_data_packet(device_id: str) -> dict:
        #     # This function would actually get the data
        #     # and could raise ConnectionError
        #     pass
        #
        # def process_iot_packet(device_id: str, pkt_data_param_perhaps_config: dict):
        #     try:
        #         actual_received_data = receive_data_packet(device_id)
        #         # then process actual_received_data, possibly using pkt_data_param_perhaps_config
        #         return actual_received_data
        #     except ConnectionError:
        #         raise

        # Let's go with the most straightforward interpretation of "implement this function to simply receive and handle the data packet"
        # and "rely on a function called receive_data_packet".
        # This implies process_iot_packet will call receive_data_packet.
        # The `pkt_data` parameter to `process_iot_packet` is a bit confusing if `receive_data_packet` also fetches data.
        # Let's assume `receive_data_packet` is the one that might fail with ConnectionError and it takes `device_id`.
        # The `pkt_data` given to `process_iot_packet` might be some metadata or configuration for the processing.
        # However, the request "return the received data as is" suggests the output of `receive_data_packet` is what's returned.

        # "The function should take a device_id as a string and pkt_data as a dictionary. It should return the received data as is."
        # This strongly implies that `pkt_data` *is* the "received data".
        # "If the packet cannot be received due to a connection issue, the function should raise a ConnectionError."
        # This implies an action *within* this function that attempts reception.
        # "The system can rely on a function called receive_data_packet, which retrieves data from devices."

        # Let's assume `receive_data_packet` is called, and it's the source of the data and potential error.
        # The `pkt_data` parameter for `process_iot_packet` is still a bit odd if it's not the data itself.
        # If `pkt_data` *is* the data, then `receive_data_packet` might have been called *before* `process_iot_packet`.
        # "Can you help implement this function to simply receive and handle the data packet?"
        # This suggests `process_iot_packet` does the receiving.

        # Simplest interpretation:
        # `process_iot_packet` calls `receive_data_packet(device_id)`
        # The `pkt_data` parameter to `process_iot_packet` is unused if `receive_data_packet` doesn't take it.
        # Or, `receive_data_packet(device_id, pkt_data)` is called.
        # "retrieves data from devices" implies `receive_data_packet` is the active part.

        # Let's assume `receive_data_packet` is the function that actually performs the I/O
        # and thus can raise ConnectionError.
        # The `pkt_data` argument to `process_iot_packet` is a bit of a red herring if `receive_data_packet`
        # is the sole source of the data to be returned.
        # If `pkt_data` is meant to be the data *payload* that `receive_data_packet` should try to get,
        # then `receive_data_packet` should probably accept it.

        # "The function should take a device_id as a string and pkt_data as a dictionary."
        # "It should return the received data as is."
        # "If the packet cannot be received ... raise a ConnectionError."
        # "rely on a function called receive_data_packet, which retrieves data from devices."

        # This implies:
        # 1. `process_iot_packet` is the main entry point.
        # 2. It uses `receive_data_packet` to get the data.
        # 3. `receive_data_packet` is the one that might raise `ConnectionError`.
        # 4. The data returned by `receive_data_packet` is then returned by `process_iot_packet`.
        # What about `pkt_data`? If `receive_data_packet` takes `device_id` to know *which* device,
        # perhaps `pkt_data` is some specific request *to* that device, or configuration for the retrieval.

        # Let's assume `receive_data_packet` takes `device_id` and potentially `pkt_data` as parameters for the request.
        # And it returns the actual data.

        # Placeholder for the external function
        # def receive_data_packet(device_id: str, request_data: dict) -> dict:
        #     # Simulates receiving data. In a real scenario, this would involve I/O.
        #     # For example, it might make a network request to the device.
        #     # If connection fails:
        #     # raise ConnectionError("Failed to connect to device")
        #     # If successful:
        #     # return {"temperature": 25, "humidity": 60} # Example data
        #     pass # This function is assumed to exist

        # The prompt says "return the received data as is". This means the data from receive_data_packet.
        # The `pkt_data` parameter for `process_iot_packet` is likely passed to `receive_data_packet`.

        data = receive_data_packet(device_id, pkt_data) # Assuming this signature for receive_data_packet
        return data
    except ConnectionError as e:
        # Log the error or handle it if necessary before re-raising
        # print(f"ConnectionError while processing packet for {device_id}: {e}")
        raise ConnectionError(f"Failed to receive packet from device {device_id}: {e}") from e

# Note: The function `receive_data_packet` is assumed to be defined elsewhere.
# For this code to be runnable, `receive_data_packet` would need to be implemented
# or mocked, and it should be capable of raising a ConnectionError.

# Example of how `receive_data_packet` might be defined (outside this request's scope):
#
# import random
# def receive_data_packet(device_id: str, request_payload: dict) -> dict:
#     """
#     Simulates retrieving a data packet from an IoT device.
#     This function is a dependency and would be provided by the system.
#     """
#     print(f"Attempting to receive data from device {device_id} with payload {request_payload}...")
#     # Simulate a chance of connection error
#     if random.random() < 0.3: # 30% chance of error
#         raise ConnectionError(f"Simulated connection error for device {device_id}")
#
#     # Simulate successful data retrieval
#     return {"device_id": device_id, "status": "ok", "data_echo": request_payload, "value": random.randint(1, 100)}
