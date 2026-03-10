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
    if patient_code not in patients:
        raise ValueError("Patient ID not found")

    patient = patients[patient_code]

    sensitive_fields = ['diagnosis', 'treatment_plan']
    for field in sensitive_fields:
        if f"{{patient.{field}}}" in report_template:
            raise ValueError(f"Attempt to access sensitive information: {field}")

    # Prepare data for formatting, only allowing non-sensitive fields
    # The template uses placeholders like {patient.name}, so we create keys matching this.
    allowed_data = {
        "patient.id": patient.id,
        "patient.name": patient.name,
        "patient.age": patient.age,
        "patient.gender": patient.gender
    }

    # A simple way to replace placeholders if they are exactly as keys in allowed_data.
    # For more complex templating, a regex-based substitution or a templating engine
    # might be more robust, but str.format() works if keys match placeholders.
    # We need to ensure that only allowed placeholders are substituted.
    # One way is to iterate and replace, or use a custom Formatter.
    # For simplicity with .format(), we can pre-process the template or use .format_map
    # with a dictionary that only contains allowed keys.

    # Let's ensure that only the allowed placeholders are in the template
    # or handle KeyError during formatting if an unexpected placeholder exists.
    # The current requirement is to raise error for sensitive fields, not for
    # general unknown fields.

    try:
        # We use format_map to avoid issues if the template contains other curly braces
        # not intended as placeholders, or placeholders not in allowed_data (which would
        # then raise a KeyError, which is acceptable if strictness is desired).
        # To be more robust against arbitrary placeholders, one might parse the template
        # and only substitute known, allowed ones.
        # For this problem, we assume `report_template.format(**allowed_data)` is sufficient
        # given the check for sensitive fields.
        report = report_template.format(**allowed_data)
    except KeyError as e:
        # This handles cases where the template contains a placeholder
        # (e.g., {patient.unknown_field}) that is not in allowed_data
        # and is not one of the explicitly forbidden sensitive fields.
        # The problem description doesn't specify behavior for this,
        # so raising an error or ignoring it are options.
        # Raising an error is safer.
        raise ValueError(f"Template contains an unsupported placeholder: {e}") from e
        
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
