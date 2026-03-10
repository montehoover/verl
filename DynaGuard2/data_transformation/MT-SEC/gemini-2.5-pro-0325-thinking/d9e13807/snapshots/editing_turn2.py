message_log = []

def log_message(message: str, sender_id: str, timestamp: str) -> bool:
    """
    Appends a message along with its metadata to the global message_log.

    Args:
        message: The string message to log.
        sender_id: The ID of the message sender.
        timestamp: The timestamp of the message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    try:
        log_entry = {
            "message": message,
            "sender_id": sender_id,
            "timestamp": timestamp
        }
        message_log.append(log_entry)
        return True
    except Exception:
        # In a real-world scenario, you might want to log the exception
        # or handle specific exceptions. For this simple case,
        # any exception during append will result in False.
        return False

if __name__ == '__main__':
    # Example Usage
    if log_message("First message logged.", "user123", "2025-05-24T10:00:00Z"):
        print("Log 1 successful.")
    else:
        print("Log 1 failed.")

    if log_message("Second message logged.", "user456", "2025-05-24T10:01:00Z"):
        print("Log 2 successful.")
    else:
        print("Log 2 failed.")

    print("\nCurrent log:")
    for entry in message_log:
        print(f"- Sender: {entry['sender_id']}, Timestamp: {entry['timestamp']}, Message: {entry['message']}")

    # Example of a potential failure (though hard to simulate with list.append)
    # For demonstration, let's imagine a scenario where it could fail
    # For now, this will always succeed.
    # To simulate a failure, one might temporarily redefine append to raise an error.
    
    # Test with non-string (though type hint suggests str, let's see behavior)
    # Note: The current implementation will attempt to append non-strings too.
    # If strict string-only logging is desired, add a type check.
    # if log_message(123): 
    #     print("Log 3 successful (non-string).")
    # else:
    #     print("Log 3 failed (non-string).")

    # print("\nFinal log:")
    # for msg in message_log:
    #     print(f"- {msg}")
