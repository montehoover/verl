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

SENSITIVE_FIELDS = ["diagnosis", "treatment_plan"]

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Applies a provided report template to generate a string detailing 
    non-sensitive information about the patient.

    Args:
        patient_uid: The identification code for the patient.
        report_schema: A string template containing fields to populate 
                       with patient details.

    Returns:
        A string with the patient's non-sensitive information formatted 
        as per the template.

    Raises:
        ValueError: If the patient ID is not found, or there is an attempt 
                    to access sensitive information.
    """
    if patient_uid not in patients:
        raise ValueError(f"Patient ID '{patient_uid}' not found.")

    patient = patients[patient_uid]

    # Check for attempts to access sensitive information
    for field in SENSITIVE_FIELDS:
        if f"{{patient.{field}}}" in report_schema:
            raise ValueError(f"Attempt to access sensitive information: {field}")

    # Populate the report with non-sensitive information
    report = report_schema
    report = report.replace("{patient.name}", str(patient.name))
    report = report.replace("{patient.age}", str(patient.age))
    report = report.replace("{patient.gender}", str(patient.gender))
    # Add other non-sensitive fields here if necessary, e.g., patient.id
    report = report.replace("{patient.id}", str(patient.id))

    return report

if __name__ == '__main__':
    # Example Usage:
    # Valid report
    try:
        report_template_valid = "Patient Report:\nName: {patient.name}\nAge: {patient.age}\nGender: {patient.gender}\nID: {patient.id}"
        valid_report = compose_patient_report("P001", report_template_valid)
        print("Generated Valid Report:")
        print(valid_report)
    except ValueError as e:
        print(f"Error generating valid report: {e}")

    print("-" * 20)

    # Report attempting to access sensitive information
    try:
        report_template_sensitive = "Patient Report:\nName: {patient.name}\nDiagnosis: {patient.diagnosis}"
        sensitive_report = compose_patient_report("P001", report_template_sensitive)
        print("\nGenerated Sensitive Report (should not happen):")
        print(sensitive_report)
    except ValueError as e:
        print(f"\nError generating sensitive report: {e}")

    print("-" * 20)

    # Report for a non-existent patient
    try:
        report_template_non_existent = "Patient Report:\nName: {patient.name}"
        non_existent_report = compose_patient_report("P002", report_template_non_existent)
        print("\nGenerated Report for Non-existent Patient (should not happen):")
        print(non_existent_report)
    except ValueError as e:
        print(f"\nError generating report for non-existent patient: {e}")
    
    print("-" * 20)

    # Report with unhandled placeholders (they will remain as is)
    try:
        report_template_unhandled = "Patient Report:\nName: {patient.name}\nBlood Type: {patient.blood_type}"
        unhandled_report = compose_patient_report("P001", report_template_unhandled)
        print("\nGenerated Report with Unhandled Placeholders:")
        print(unhandled_report)
    except ValueError as e:
        print(f"Error generating report with unhandled placeholders: {e}")
