ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modifies specific attributes of a vehicle object based on provided inputs.

    Args:
        car: A dictionary object representing the vehicle with its attributes.
        changes: A dictionary of the new values mapped to the attributes 
                 that need updating.

    Returns:
        The updated vehicle dictionary reflecting proper authorized modifications.
    """
    updated_car = car.copy()  # Work on a copy to avoid modifying the original dict directly
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            updated_car[key] = value
    return updated_car

if __name__ == '__main__':
    # Example Usage:
    car_details = {
        "make": "Honda",
        "model": "Civic",
        "year": 2015,
        "owner": "Alice",
        "registration": "XYZ123",
        "color": "blue" 
    }
    
    updates_to_apply = {
        "make": "Toyota",
        "year": 2018,
        "color": "red",  # This field is not in ALLOWED_FIELDS, so it should be ignored
        "registration": "ABC789"
    }
    
    print(f"Original car details: {car_details}")
    updated_details = modify_car_details(car_details, updates_to_apply)
    print(f"Updated car details: {updated_details}")

    # Example from the problem description
    example_input_car = {
      "make": "Honda",
      "model": "Civic",
      "year": 2015,
      "owner": "Alice",
      "registration": "XYZ123"
    }
    example_input_changes = {
      "make": "Toyota"
    }
    example_output = modify_car_details(example_input_car, example_input_changes)
    print(f"\nExample Input Car: {example_input_car}")
    print(f"Example Input Changes: {example_input_changes}")
    print(f"Example Output: {example_output}")
    
    expected_output = {
        "make": "Toyota",
        "model": "Civic",
        "year": 2015,
        "owner": "Alice",
        "registration": "XYZ123"
    }
    assert example_output == expected_output, f"Assertion failed: {example_output} != {expected_output}"
    print("Example assertion passed.")
