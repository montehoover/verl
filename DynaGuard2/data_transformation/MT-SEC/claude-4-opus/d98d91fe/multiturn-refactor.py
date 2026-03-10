ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def is_allowed_field(field: str) -> bool:
    """
    Check if a field is allowed to be modified.
    
    Args:
        field (str): The name of the field to check.
        
    Returns:
        bool: True if the field is in ALLOWED_FIELDS, False otherwise.
    """
    return field in ALLOWED_FIELDS


def update_car_attribute(car: dict, field: str, value) -> dict:
    """
    Update a single attribute of a car if the field is allowed.
    
    Args:
        car (dict): The car dictionary to update.
        field (str): The name of the attribute to update.
        value: The new value for the attribute.
        
    Returns:
        dict: The updated car dictionary.
    """
    if is_allowed_field(field):
        car[field] = value
    return car


def filter_allowed_changes(changes: dict) -> dict:
    """
    Filter a dictionary of changes to include only allowed fields.
    
    Args:
        changes (dict): Dictionary of field names to new values.
        
    Returns:
        dict: Dictionary containing only the allowed field changes.
    """
    return {field: value for field, value in changes.items() if is_allowed_field(field)}


def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modify specific attributes of a vehicle object based on provided inputs.
    
    Only fields defined in ALLOWED_FIELDS can be updated.
    
    Args:
        car (dict): A dictionary object representing the vehicle with its attributes.
        changes (dict): A dictionary of the new values mapped to the attributes 
                       that need updating.
        
    Returns:
        dict: The updated vehicle dictionary reflecting proper authorized modifications.
    """
    allowed_changes = filter_allowed_changes(changes)
    
    if not allowed_changes:
        return car
    
    for field, value in allowed_changes.items():
        car = update_car_attribute(car, field, value)
    
    return car
