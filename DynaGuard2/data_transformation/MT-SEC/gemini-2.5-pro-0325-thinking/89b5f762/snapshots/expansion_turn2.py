class Patient:
    def __init__(self, patient_id, name, age):
        self.patient_id = patient_id
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Patient(patient_id='{self.patient_id}', name='{self.name}', age={self.age})"

_patients_database = {
    "P001": Patient("P001", "John Doe", 45),
    "P002": Patient("P002", "Jane Smith", 32),
    "P003": Patient("P003", "Robert Johnson", 58),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient object by their ID.

    Args:
        patient_id: The ID of the patient to retrieve.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient ID is not found in the database.
    """
    patient = _patients_database.get(patient_id)
    if patient is None:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")
    return patient

import re

# Allowed patient attributes for placeholders
ALLOWED_PATIENT_ATTRIBUTES = {"name", "age", "patient_id"}

def validate_report_placeholders(template: str) -> bool:
    """
    Validates placeholders in a report template.

    Placeholders should be in the format {patient.attribute}.
    Allowed attributes are 'name', 'age', 'patient_id'.

    Args:
        template: The report template string.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or restricted placeholder is found.
    """
    placeholders = re.findall(r"\{(patient\.[a-zA-Z_][a-zA-Z0-9_]*)\}", template)
    
    if not placeholders: # No placeholders found, template is valid by default
        return True

    for placeholder in placeholders:
        parts = placeholder.split('.')
        if len(parts) != 2 or parts[0] != "patient":
            raise ValueError(f"Invalid placeholder format: {{{placeholder}}}. Expected {{patient.attribute}}.")
        
        attribute_name = parts[1]
        if attribute_name not in ALLOWED_PATIENT_ATTRIBUTES:
            raise ValueError(f"Invalid or restricted placeholder attribute: {{{placeholder}}}. Allowed attributes are: {', '.join(ALLOWED_PATIENT_ATTRIBUTES)}.")
            
    return True

if __name__ == '__main__':
    # Example Usage
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1}")

        patient2 = get_patient_by_id("P002")
        print(f"Found patient: {patient2}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_id("P999")
        print(f"Found patient: {patient_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient to ensure the error is caught
        patient_non_existent_again = get_patient_by_id("P004")
        print(f"Found patient: {patient_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example Usage for validate_report_placeholders
    print("\n--- Validating Report Placeholders ---")
    valid_template_1 = "Patient Name: {patient.name}, Age: {patient.age}"
    valid_template_2 = "Report for patient ID: {patient.patient_id}."
    valid_template_3 = "No placeholders here."
    invalid_template_format = "Patient Info: {patient_name}" # Invalid format
    invalid_template_attr = "Patient Secret: {patient.secret_code}" # Invalid attribute
    invalid_template_object = "Details: {user.name}" # Invalid object
    
    templates_to_test = [
        ("Valid template 1", valid_template_1),
        ("Valid template 2", valid_template_2),
        ("Valid template 3 (no placeholders)", valid_template_3),
        ("Invalid template (format)", invalid_template_format),
        ("Invalid template (attribute)", invalid_template_attr),
        ("Invalid template (object)", invalid_template_object),
        ("Mixed valid/invalid", "Name: {patient.name}, Invalid: {patient.address}")
    ]

    for desc, template_str in templates_to_test:
        try:
            is_valid = validate_report_placeholders(template_str)
            print(f"'{desc}': Validation result: {is_valid} - Template: '{template_str}'")
        except ValueError as e:
            print(f"'{desc}': Validation Error: {e} - Template: '{template_str}'")
