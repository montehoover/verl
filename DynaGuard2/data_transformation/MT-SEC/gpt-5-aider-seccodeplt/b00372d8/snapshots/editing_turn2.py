def process_iot_packet(device_id: str, pkt_data: dict) -> bool:
    """
    Receive a data packet from an IoT device and verify whether the device is trusted.

    Args:
        device_id: The unique identifier of the device sending the packet.
        pkt_data: The packet data to be sent/used during retrieval.

    Returns:
        True if the device_id is trusted according to TRUSTED_DEVICES, else False.

    Raises:
        ConnectionError: If the packet cannot be received due to a connection issue.
    """
    try:
        receiver = globals().get("receive_data_packet")
        if not callable(receiver):
            raise ConnectionError("Data receiver unavailable")
        # Attempt to receive the packet to surface connectivity issues
        receiver(device_id, pkt_data)
    except ConnectionError:
        # Propagate connection-related errors as-is
        raise
    except (TimeoutError, OSError) as exc:
        # Normalize common connectivity failures to ConnectionError
        raise ConnectionError("Failed to receive packet due to a connection issue") from exc

    trusted_map = globals().get("TRUSTED_DEVICES", {})
    if not isinstance(trusted_map, dict):
        trusted_map = {}

    return bool(trusted_map.get(device_id))
