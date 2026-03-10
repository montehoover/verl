class Patient:
    """Represents a patient."""
    def __init__(self, code, name, age):
        self.code = code
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Patient(code='{self.code}', name='{self.name}', age={self.age})"

_patients_database = {
    "P001": Patient("P001", "John Doe", 45),
    "P002": Patient("P002", "Jane Smith", 32),
    "P003": Patient("P003", "Alice Brown", 58),
}

def get_patient_by_code(patient_code: str) -> Patient:
    """
    Retrieves a patient object from the predefined dictionary using their code.

    Args:
        patient_code: The unique code of the patient.

    Returns:
        The Patient object corresponding to the given code.

    Raises:
        ValueError: If the patient code does not exist in the database.
    """
    patient = _patients_database.get(patient_code)
    if patient is None:
        raise ValueError(f"Patient with code '{patient_code}' not found.")
    return patient

import re

_ALLOWED_PLACEHOLDERS = {
    "patient.name",
    "patient.age",
}

def validate_report_template(template_string: str) -> bool:
    """
    Validates a report template string to ensure it only uses allowed, non-sensitive placeholders.

    Args:
        template_string: The report template string with placeholders like {placeholder}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is found.
    """
    placeholders = re.findall(r"\{(.*?)\}", template_string)
    for placeholder in placeholders:
        if placeholder not in _ALLOWED_PLACEHOLDERS:
            raise ValueError(
                f"Invalid or sensitive placeholder found: '{placeholder}'. "
                f"Allowed placeholders are: {', '.join(_ALLOWED_PLACEHOLDERS)}"
            )
    return True

if __name__ == '__main__':
    # Example usage:
    try:
        patient1 = get_patient_by_code("P001")
        print(f"Found patient: {patient1}")

        patient2 = get_patient_by_code("P002")
        print(f"Found patient: {patient2}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_code("P999")
        print(f"Found patient: {patient_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example usage for validate_report_template:
    print("\n--- Template Validation ---")
    valid_template = "Patient Report:\nName: {patient.name}\nAge: {patient.age}"
    invalid_template_1 = "Patient Report:\nName: {patient.name}\nCode: {patient.code}" # patient.code is sensitive
    invalid_template_2 = "Patient Report:\nDetails: {patient.details}" # patient.details is unknown

    try:
        print(f"Validating template: \"{valid_template}\"")
        validate_report_template(valid_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        print(f"\nValidating template: \"{invalid_template_1}\"")
        validate_report_template(invalid_template_1)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        print(f"\nValidating template: \"{invalid_template_2}\"")
        validate_report_template(invalid_template_2)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        # Another non-existent patient to show the error again
        patient_non_existent_again = get_patient_by_code("P004")
        print(f"Found patient: {patient_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")
