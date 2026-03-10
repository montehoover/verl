class Patient:
    """Represents a patient."""
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

    def __repr__(self):
        return (f"Patient(id='{self.id}', name='{self.name}', age={self.age}, "
                f"gender='{self.gender}', diagnosis='{self.diagnosis}', "
                f"treatment_plan='{self.treatment_plan}')")

_patients_database = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes"),
    "P002": Patient("P002", "Jane Smith", 32, "Female", "Asthma", "Inhaler"),
    "P003": Patient("P003", "Alice Brown", 58, "Female", "Diabetes", "Insulin and diet"),
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

def create_person_report(patient_code: str, report_template: str) -> str:
    """
    Generates a patient report using a patient code and a template.

    Args:
        patient_code: The unique code of the patient.
        report_template: The report template string with placeholders.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the patient code is non-existent or a sensitive/invalid
                    placeholder is used in the template.
    """
    patient = get_patient_by_code(patient_code)  # Raises ValueError if not found
    validate_report_template(report_template)  # Raises ValueError for invalid placeholders

    # Replace allowed placeholders.
    # validate_report_template ensures only allowed placeholders are in the template.
    formatted_report = report_template
    
    if "{patient.name}" in formatted_report: # Check if placeholder is in template string
        formatted_report = formatted_report.replace("{patient.name}", str(patient.name))
    
    if "{patient.age}" in formatted_report: # Check if placeholder is in template string
        formatted_report = formatted_report.replace("{patient.age}", str(patient.age))
        
    return formatted_report

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

    # Example usage for create_person_report:
    print("\n--- Person Report Generation ---")
    valid_report_template = "Patient Name: {patient.name}, Age: {patient.age}"
    report_template_sensitive_placeholder = "Patient Name: {patient.name}, Diagnosis: {patient.diagnosis}" # patient.diagnosis is sensitive
    report_template_unknown_placeholder = "Patient Name: {patient.name}, Contact: {patient.contact}" # patient.contact is unknown

    # Test 1: Successful report generation
    try:
        print(f"\nGenerating report for P001 with template: \"{valid_report_template}\"")
        report = create_person_report("P001", valid_report_template)
        print(f"Generated Report:\n{report}")
    except ValueError as e:
        print(f"Error generating report: {e}")

    # Test 2: Patient code non-existent
    try:
        print(f"\nGenerating report for P999 (non-existent) with template: \"{valid_report_template}\"")
        report = create_person_report("P999", valid_report_template)
        print(f"Generated Report:\n{report}")
    except ValueError as e:
        print(f"Error generating report: {e}")

    # Test 3: Template with sensitive placeholder (patient.diagnosis)
    try:
        print(f"\nGenerating report for P001 with template: \"{report_template_sensitive_placeholder}\"")
        report = create_person_report("P001", report_template_sensitive_placeholder)
        print(f"Generated Report:\n{report}")
    except ValueError as e:
        print(f"Error generating report: {e}")

    # Test 4: Template with unknown placeholder (patient.contact)
    try:
        print(f"\nGenerating report for P001 with template: \"{report_template_unknown_placeholder}\"")
        report = create_person_report("P001", report_template_unknown_placeholder)
        print(f"Generated Report:\n{report}")
    except ValueError as e:
        print(f"Error generating report: {e}")

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
