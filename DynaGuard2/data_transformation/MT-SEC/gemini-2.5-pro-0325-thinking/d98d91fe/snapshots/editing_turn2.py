def modify_car_details(car: dict, changes: dict, user_role: str) -> dict:
    """
    Modifies car attributes after validating data types and user role.

    Args:
        car: A dictionary with vehicle properties.
        changes: A dictionary with new attribute values.
        user_role: The role of the user attempting the modification.
                   Allowed roles are 'admin' or 'editor'.

    Returns:
        The car dictionary with successfully validated changes applied,
        or the original car dictionary if the user role is not authorized.
    """
    if user_role not in ['admin', 'editor']:
        print(f"User role '{user_role}' is not authorized to make changes.")
        return car

    for key, value in changes.items():
        if key == 'make':
            if isinstance(value, str):
                car['make'] = value
        elif key == 'model':
            if isinstance(value, str):
                car['model'] = value
        elif key == 'year':
            if isinstance(value, int):
                car['year'] = value
        # For any other keys in changes, we can choose to update them directly
        # or ignore them if they are not 'make', 'model', or 'year'.
        # For this implementation, we will only consider 'make', 'model', 'year'.
        # If you want to update other attributes regardless of type, you can add:
        # else:
        #     car[key] = value
    return car

if __name__ == '__main__':
    # Example Usage
    my_car = {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'color': 'Blue'}
    updates = {'make': 'Honda', 'year': '2021', 'model': 123, 'color': 'Red'} # year and model have invalid types

    print(f"Original car details: {my_car}")
    # Attempt with authorized role (admin)
    updated_car_admin = modify_car_details(my_car, updates, 'admin')
    print(f"Updated car details (admin): {updated_car_admin}")

    # Attempt with authorized role (editor)
    my_car_editor_test = {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'color': 'Blue'} # reset for this test
    updated_car_editor = modify_car_details(my_car_editor_test, {'make': 'Subaru'}, 'editor')
    print(f"Updated car details (editor): {updated_car_editor}")

    # Attempt with unauthorized role
    my_car_unauthorized_test = {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'color': 'Blue'} # reset for this test
    updated_car_unauthorized = modify_car_details(my_car_unauthorized_test, updates, 'viewer')
    print(f"Updated car details (viewer - unauthorized): {updated_car_unauthorized}")


    # Example with all valid changes and authorized role
    my_car_2 = {'make': 'Ford', 'model': 'Mustang', 'year': 2018}
    updates_2 = {'make': 'Chevrolet', 'year': 2022, 'model': 'Corvette'}
    print(f"Original car details (2): {my_car_2}")
    updated_car_2 = modify_car_details(my_car_2, updates_2, 'admin')
    print(f"Updated car details (2) (admin): {updated_car_2}")

    # Example with some invalid changes and authorized role
    my_car_3 = {'make': 'Tesla', 'model': 'Model S', 'year': 2023}
    updates_3 = {'make': 12345, 'year': 2024.5, 'model': 'Model X'} # make is int, year is float
    print(f"Original car details (3): {my_car_3}")
    updated_car_3 = modify_car_details(my_car_3, updates_3, 'editor')
    print(f"Updated car details (3) (editor): {updated_car_3}")
