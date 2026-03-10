import logging

# Configure basic logging for demonstration purposes
# In a real application, this would likely be configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def _is_field_allowed(field: str, allowed_fields_list: list) -> bool:
    """Checks if a field is in the list of allowed fields."""
    return field in allowed_fields_list

def _update_single_car_attribute(car: dict, field: str, value: any) -> None:
    """Updates a single attribute of the car dictionary in place."""
    car[field] = value

def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Updates specific fields of a vehicle's attributes using given input data.

    Args:
        car: dict, representing the current state of vehicle attributes.
        updates: dict, which contains the fields and new values for updating the vehicle.

    Returns:
        The updated vehicle dictionary, reflecting any valid modifications.
    """
    for key, new_value in updates.items():
        if _is_field_allowed(key, ALLOWED_FIELDS):
            original_value = car.get(key)
            _update_single_car_attribute(car, key, new_value)
            logger.info(f"Updated attribute '{key}': original_value='{original_value}', new_value='{new_value}'")
    return car
