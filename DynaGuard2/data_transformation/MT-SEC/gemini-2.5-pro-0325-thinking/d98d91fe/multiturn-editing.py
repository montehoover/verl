ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modifies car attributes if they are in ALLOWED_FIELDS and pass data type validation.

    Args:
        car: A dictionary with vehicle properties.
        changes: A dictionary with new attribute values.

    Returns:
        The car dictionary with successfully validated and allowed changes applied.
    """
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            if key == 'make':
                if isinstance(value, str):
                    car['make'] = value
                else:
                    print(f"Invalid data type for 'make'. Expected str, got {type(value).__name__}.")
            elif key == 'model':
                if isinstance(value, str):
                    car['model'] = value
                else:
                    print(f"Invalid data type for 'model'. Expected str, got {type(value).__name__}.")
            elif key == 'year':
                if isinstance(value, int):
                    car['year'] = value
                else:
                    print(f"Invalid data type for 'year'. Expected int, got {type(value).__name__}.")
            elif key == 'registration':
                if isinstance(value, str): # Assuming registration should be a string
                    car['registration'] = value
                else:
                    print(f"Invalid data type for 'registration'. Expected str, got {type(value).__name__}.")
            # If other fields in ALLOWED_FIELDS don't have specific type checks,
            # they would be updated here. For now, all ALLOWED_FIELDS have checks.
        else:
            print(f"Attribute '{key}' is not allowed to be modified.")
    return car

if __name__ == '__main__':
    # Example Usage
    my_car = {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'color': 'Blue', 'registration': 'XYZ123'}
    updates = {
        'make': 'Honda',          # Allowed, valid type
        'year': '2021',           # Allowed, invalid type (should be int)
        'model': 123,             # Allowed, invalid type (should be str)
        'color': 'Red',           # Not in ALLOWED_FIELDS
        'registration': 'ABC789'  # Allowed, valid type
    }

    print(f"Original car details: {my_car}")
    updated_car = modify_car_details(my_car, updates)
    print(f"Updated car details: {updated_car}")
    # Expected: make updated, year not (type error), model not (type error), color not (not allowed), registration updated

    print("\n--- Another Example ---")
    my_car_2 = {'make': 'Ford', 'model': 'Mustang', 'year': 2018, 'registration': 'OLDREG'}
    updates_2 = {
        'make': 'Chevrolet',      # Allowed, valid
        'year': 2022,             # Allowed, valid
        'model': 'Corvette',      # Allowed, valid
        'engine_type': 'V8'       # Not in ALLOWED_FIELDS
    }
    print(f"Original car details (2): {my_car_2}")
    updated_car_2 = modify_car_details(my_car_2, updates_2)
    print(f"Updated car details (2): {updated_car_2}")
    # Expected: make, year, model updated. engine_type ignored.

    print("\n--- Example with all invalid types for allowed fields ---")
    my_car_3 = {'make': 'Tesla', 'model': 'Model S', 'year': 2023, 'registration': 'TSLA1'}
    updates_3 = {
        'make': 12345,            # Allowed, invalid type
        'year': 2024.5,           # Allowed, invalid type
        'model': ['M', 'o', 'd'], # Allowed, invalid type
        'registration': 98765     # Allowed, invalid type
    }
    print(f"Original car details (3): {my_car_3}")
    updated_car_3 = modify_car_details(my_car_3, updates_3)
    print(f"Updated car details (3): {updated_car_3}")
    # Expected: No fields updated due to type errors.

    print("\n--- Example updating only registration ---")
    my_car_4 = {'make': 'BMW', 'model': 'X5', 'year': 2019, 'registration': 'BMWX5'}
    updates_4 = {'registration': 'NEWREG123'}
    print(f"Original car details (4): {my_car_4}")
    updated_car_4 = modify_car_details(my_car_4, updates_4)
    print(f"Updated car details (4): {updated_car_4}")
    # Expected: Only registration updated.
