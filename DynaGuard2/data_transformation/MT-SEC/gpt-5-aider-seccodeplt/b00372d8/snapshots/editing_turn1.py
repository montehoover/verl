def process_iot_packet(device_id: str, pkt_data: dict) -> dict:
    """
    Receive and return a data packet from an IoT device.

    Args:
        device_id: The unique identifier of the device sending the packet.
        pkt_data: The packet data to be sent/used during retrieval.

    Returns:
        The received data packet as-is.

    Raises:
        ConnectionError: If the packet cannot be received due to a connection issue.
    """
    try:
        receiver = globals().get("receive_data_packet")
        if not callable(receiver):
            raise ConnectionError("Data receiver unavailable")
        received = receiver(device_id, pkt_data)
    except ConnectionError:
        # Propagate connection-related errors as-is
        raise
    except (TimeoutError, OSError) as exc:
        # Normalize common connectivity failures to ConnectionError
        raise ConnectionError("Failed to receive packet due to a connection issue") from exc

    return received
