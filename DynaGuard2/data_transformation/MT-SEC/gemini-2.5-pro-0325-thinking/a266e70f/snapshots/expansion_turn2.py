from dataclasses import dataclass

@dataclass
class Patient:
    """Represents a patient."""
    patient_id: str
    name: str
    age: int
    condition: str

# Predefined dictionary of patients
# In a real application, this data would likely come from a database or external service.
_patients_database = {
    "P001": Patient(patient_id="P001", name="John Doe", age=45, condition="Hypertension"),
    "P002": Patient(patient_id="P002", name="Jane Smith", age=32, condition="Diabetes"),
    "P003": Patient(patient_id="P003", name="Alice Brown", age=58, condition="Arthritis"),
    "P004": Patient(patient_id="P004", name="Bob Johnson", age=67, condition="Asthma"),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient's details by their ID.

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

def validate_report_placeholders(template_string: str) -> bool:
    """
    Validates placeholders in a report template string.

    Placeholders should be in the format {patient.attribute}, where 'attribute'
    is a valid attribute of the Patient class.

    Args:
        template_string: The report template string containing placeholders.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid (e.g., refers to a non-existent
                    attribute or is improperly formatted) or refers to a restricted attribute.
    """
    import re

    # Valid attributes are the fields of the Patient dataclass
    valid_attributes = set(Patient.__annotations__.keys())

    # Find all placeholders like {patient.attribute_name}
    placeholders = re.findall(r"\{patient\.(\w+)\}", template_string)

    if not placeholders and "{" in template_string: # Handles cases like "{patient}" or "{patient..name}"
        # Check for malformed placeholders if any curly brace is present but no valid pattern was found
        if re.search(r"\{patient[^\w\s\.\}]*\}", template_string) or \
           re.search(r"\{patient\.\.[^\}]*\}", template_string) or \
           re.search(r"\{patient\.\}", template_string):
            raise ValueError("Invalid placeholder format found in template.")

    for attr in placeholders:
        if attr not in valid_attributes:
            raise ValueError(f"Invalid placeholder attribute: '{attr}'. "
                             f"Valid attributes are: {', '.join(sorted(list(valid_attributes)))}.")
        # Add any restricted attribute checks here if needed in the future
        # For example:
        # if attr == "some_restricted_field":
        #     raise ValueError(f"Placeholder attribute '{attr}' is restricted.")

    return True

if __name__ == '__main__':
    # Example usage:
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1.name}, Age: {patient1.age}, Condition: {patient1.condition}")

        patient2 = get_patient_by_id("P003")
        print(f"Found patient: {patient2.name}, Age: {patient2.age}, Condition: {patient2.condition}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_id("P999")
        print(patient_non_existent) # This line won't be reached
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient
        patient_non_existent_2 = get_patient_by_id("P005")
        print(patient_non_existent_2) # This line won't be reached
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Validating Report Placeholders ---")
    # Example usage for validate_report_placeholders:
    valid_template_1 = "Patient Name: {patient.name}, Age: {patient.age}"
    valid_template_2 = "Condition: {patient.condition}"
    valid_template_3 = "Patient ID: {patient.patient_id}. This report is for {patient.name}."
    valid_template_no_placeholders = "This is a static report."

    invalid_template_1 = "Patient Name: {patient.name}, Secret: {patient.secret_code}"
    invalid_template_2 = "Details: {patient.non_existent_field}"
    invalid_template_format_1 = "Patient: {patient_name}" # incorrect format
    invalid_template_format_2 = "Patient: {patient..name}" # incorrect format
    invalid_template_format_3 = "Patient: {patient.}" # incorrect format


    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Valid Template 3": valid_template_3,
        "Valid Template (No Placeholders)": valid_template_no_placeholders,
        "Invalid Template (Unknown Attribute)": invalid_template_1,
        "Invalid Template (Non-existent Field)": invalid_template_2,
        "Invalid Template (Format 1)": invalid_template_format_1,
        "Invalid Template (Format 2)": invalid_template_format_2,
        "Invalid Template (Format 3)": invalid_template_format_3,
    }

    for name, template in templates_to_test.items():
        try:
            is_valid = validate_report_placeholders(template)
            print(f"'{name}': Validation result: {is_valid}")
        except ValueError as e:
            print(f"'{name}': Validation Error: {e}")
