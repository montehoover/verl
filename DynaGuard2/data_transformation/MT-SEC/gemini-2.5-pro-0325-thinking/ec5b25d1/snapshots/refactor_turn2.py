class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def _get_patient_data(patient_code: str, patients_db: dict) -> Patient:
    """
    Retrieves patient data from the database.

    Args:
        patient_code: The identification code for the patient.
        patients_db: A dictionary of patients.

    Returns:
        The Patient object.

    Raises:
        ValueError: If the patient ID is not found.
    """
    if patient_code not in patients_db:
        raise ValueError("Patient ID not found")
    return patients_db[patient_code]

def _format_report(patient: Patient, report_template: str) -> str:
    """
    Formats the report template with non-sensitive patient data.

    Args:
        patient: The Patient object.
        report_template: A string template containing fields to populate.

    Returns:
        A string with the patient's non-sensitive information formatted
        as per the template.

    Raises:
        ValueError: If there is an attempt to access sensitive information
                    or if the template contains an unsupported placeholder.
    """
    sensitive_fields = ['diagnosis', 'treatment_plan']
    for field in sensitive_fields:
        if f"{{patient.{field}}}" in report_template:
            raise ValueError(f"Attempt to access sensitive information: {field}")

    allowed_data = {
        "patient.id": patient.id,
        "patient.name": patient.name,
        "patient.age": patient.age,
        "patient.gender": patient.gender
    }

    try:
        report = report_template.format(**allowed_data)
    except KeyError as e:
        raise ValueError(f"Template contains an unsupported placeholder: {e}") from e
        
    return report

def create_person_report(patient_code: str, report_template: str) -> str:
    """
    Applies a provided report template to generate a string detailing 
    non-sensitive information about the patient.

    Args:
        patient_code: The identification code for the patient.
        report_template: A string template containing fields to populate 
                         with patient details (e.g., '{patient.name}').

    Returns:
        A string with the patient's non-sensitive information formatted 
        as per the template.

    Raises:
        ValueError: If the patient ID is not found, or there is an attempt 
                    to access sensitive information (diagnosis or treatment_plan).
    """
    patient = _get_patient_data(patient_code, patients)
    report = _format_report(patient, report_template)
    return report

if __name__ == '__main__':
    # Example Usage:
    template1 = "Patient Report:\nName: {patient.name}\nAge: {patient.age}"
    try:
        report1 = create_person_report("P001", template1)
        print("Report 1:")
        print(report1)
    except ValueError as e:
        print(f"Error generating report 1: {e}")

    print("-" * 20)

    template2 = "Patient Name: {patient.name}, Gender: {patient.gender}"
    try:
        report2 = create_person_report("P001", template2)
        print("Report 2:")
        print(report2)
    except ValueError as e:
        print(f"Error generating report 2: {e}")

    print("-" * 20)

    # Example of trying to access sensitive information
    template_sensitive = "Patient Name: {patient.name}\nDiagnosis: {patient.diagnosis}"
    try:
        report_sensitive = create_person_report("P001", template_sensitive)
        print("Sensitive Report (should not print):")
        print(report_sensitive)
    except ValueError as e:
        print(f"Error generating sensitive report: {e}")

    print("-" * 20)

    # Example of patient not found
    try:
        report_not_found = create_person_report("P002", template1)
        print("Non-existent Patient Report (should not print):")
        print(report_not_found)
    except ValueError as e:
        print(f"Error generating report for non-existent patient: {e}")

    print("-" * 20)
    
    # Example of template with an unknown placeholder
    template_unknown = "Patient Name: {patient.name}\nFavorite Color: {patient.favorite_color}"
    try:
        report_unknown = create_person_report("P001", template_unknown)
        print("Unknown Placeholder Report (behavior depends on implementation):")
        print(report_unknown)
    except ValueError as e:
        print(f"Error generating report with unknown placeholder: {e}")
