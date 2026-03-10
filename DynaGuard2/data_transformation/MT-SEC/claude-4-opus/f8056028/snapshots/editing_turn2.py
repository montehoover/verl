def split_log_entry(log_entry):
    """
    Split a log entry into its components: timestamp, log level, and message.
    
    Args:
        log_entry (str): A log entry string in format "YYYY-MM-DD HH:MM:SS [LEVEL] Message"
    
    Returns:
        list: A list containing [timestamp, log_level, message]
    """
    # Find the first space after the date
    first_space = log_entry.find(' ')
    if first_space == -1:
        return [log_entry, '', '']
    
    # Find the second space after the time
    second_space = log_entry.find(' ', first_space + 1)
    if second_space == -1:
        return [log_entry[:first_space], log_entry[first_space+1:], '']
    
    # Extract timestamp (date and time)
    timestamp = log_entry[:second_space]
    
    # Find the log level enclosed in brackets
    bracket_start = log_entry.find('[', second_space)
    bracket_end = log_entry.find(']', bracket_start)
    
    if bracket_start == -1 or bracket_end == -1:
        # No brackets found, treat rest as message
        return [timestamp, '', log_entry[second_space+1:].strip()]
    
    # Extract log level (including brackets)
    log_level = log_entry[bracket_start:bracket_end+1]
    
    # Extract message (everything after the closing bracket)
    message = log_entry[bracket_end+1:].strip()
    
    return [timestamp, log_level, message]


def identify_log_parts(log_entry):
    """
    Identify and extract log entry components into a dictionary.
    
    Args:
        log_entry (str): A log entry string in format "YYYY-MM-DD HH:MM:SS [LEVEL] Message"
    
    Returns:
        dict: A dictionary with keys 'timestamp', 'log_level', and 'message'
    """
    # Initialize result dictionary
    result = {'timestamp': '', 'log_level': '', 'message': ''}
    
    # Find the first space after the date
    first_space = log_entry.find(' ')
    if first_space == -1:
        result['timestamp'] = log_entry
        return result
    
    # Find the second space after the time
    second_space = log_entry.find(' ', first_space + 1)
    if second_space == -1:
        result['timestamp'] = log_entry
        return result
    
    # Extract timestamp (date and time)
    result['timestamp'] = log_entry[:second_space]
    
    # Find the log level enclosed in brackets
    bracket_start = log_entry.find('[', second_space)
    bracket_end = log_entry.find(']', bracket_start)
    
    if bracket_start == -1 or bracket_end == -1:
        # No brackets found, treat rest as message
        result['message'] = log_entry[second_space+1:].strip()
        return result
    
    # Extract log level (without brackets)
    result['log_level'] = log_entry[bracket_start+1:bracket_end]
    
    # Extract message (everything after the closing bracket)
    # Check if there's a colon after the brackets
    colon_pos = log_entry.find(':', bracket_end)
    if colon_pos != -1 and colon_pos - bracket_end < 3:
        result['message'] = log_entry[colon_pos+1:].strip()
    else:
        result['message'] = log_entry[bracket_end+1:].strip()
    
    return result
