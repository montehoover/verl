class Patient:
    """
    Represents a patient with their medical information.

    Attributes:
        id (str): The unique identifier for the patient.
        name (str): The name of the patient.
        age (int): The age of the patient.
        gender (str): The gender of the patient.
        diagnosis (str): The diagnosis for the patient (sensitive information).
        treatment_plan (str): The treatment plan for the patient (sensitive information).
    """
    def __init__(self, id: str, name: str, age: int, gender: str, diagnosis: str, treatment_plan: str):
        """
        Initializes a new Patient instance.

        Args:
            id: The unique identifier for the patient.
            name: The name of the patient.
            age: The age of the patient.
            gender: The gender of the patient.
            diagnosis: The diagnosis for the patient.
            treatment_plan: The treatment plan for the patient.
        """
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Global in-memory "database" of patients.
# In a real application, this would be an actual database connection.
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


def _get_patient_data(patient_code: str, patients_db: dict) -> Patient:
    """
    Retrieves a patient's record from the provided patient database.

    This is a helper function designed to encapsulate the data retrieval logic.
    It checks if the patient exists and returns the Patient object if found.

    Args:
        patient_code: The identification code for the patient.
        patients_db: A dictionary of patients.

    Returns:
        The Patient object corresponding to the given patient_code.

    Raises:
        ValueError: If the patient_code does not exist in the patients_db.
    """
    # Check if the patient code exists in the database.
    if patient_code not in patients_db:
        # Raise an error if the patient is not found.
        raise ValueError("Patient ID not found")
    return patients_db[patient_code]


def _format_report(patient: Patient, report_template: str) -> str:
    """
    Formats a report string using a template and non-sensitive patient data.

    This function populates a given template string with placeholders
    (e.g., '{patient.name}') using the attributes of the provided Patient object.
    It explicitly disallows access to sensitive fields like 'diagnosis' and
    'treatment_plan'.

    Args:
        patient: The Patient object.
        report_template: A string template containing fields to populate.

    Returns:
        A string representing the formatted report with patient's non-sensitive
        information.

    Raises:
        ValueError: If the report_template attempts to access sensitive patient
                    information (e.g., diagnosis, treatment_plan) or if the
                    template contains an unsupported placeholder.
    """
    # Define a list of sensitive fields that should not be exposed in the report.
    sensitive_fields = ['diagnosis', 'treatment_plan']
    for field in sensitive_fields:
        # Check if any sensitive field placeholder is present in the template.
        if f"{{patient.{field}}}" in report_template:
            raise ValueError(f"Attempt to access sensitive information: {field}")

    # Prepare a dictionary with allowed non-sensitive patient data for formatting.
    # The keys in this dictionary must match the placeholders in the template string.
    allowed_data = {
        "patient.id": patient.id,
        "patient.name": patient.name,
        "patient.age": patient.age,
        "patient.gender": patient.gender
    }

    try:
        # Use the format method to substitute placeholders with actual data.
        # The **allowed_data syntax unpacks the dictionary into keyword arguments.
        report = report_template.format(**allowed_data)
    except KeyError as e:
        # If the template contains a placeholder not present in allowed_data
        # (and not a sensitive field, which is checked above), raise an error.
        raise ValueError(f"Template contains an unsupported placeholder: {e}") from e
        
    return report


def create_person_report(patient_code: str, report_template: str) -> str:
    """
    Generates a formatted report string for a given patient using a template.

    This is the main function that orchestrates the retrieval of patient data
    and the formatting of the report. It ensures that only non-sensitive
    information is used.

    Args:
        patient_code: The unique identification code for the patient.
        report_template: A string that serves as a template for the report.
                         It can contain placeholders like '{patient.name}' or
                         '{patient.age}' that will be replaced with actual
                         patient data.

    Returns:
        A string containing the formatted report with the patient's
        non-sensitive information.

    Raises:
        ValueError: If the patient_code is not found in the patient database,
                    or if the report_template attempts to access sensitive
                    information (like diagnosis or treatment_plan), or if
                    the template contains an unsupported placeholder.
    """
    # Step 1: Retrieve patient data using the helper function.
    # This can raise a ValueError if the patient is not found.
    patient = _get_patient_data(patient_code, patients)
    
    # Step 2: Format the report using the retrieved patient data and the template.
    # This can raise a ValueError for sensitive data access or unsupported placeholders.
    report = _format_report(patient, report_template)
    
    return report


if __name__ == '__main__':
    # This block demonstrates the usage of the create_person_report function
    # and its error handling capabilities.

    # Example 1: Successful report generation
    template1 = "Patient Report:\nName: {patient.name}\nAge: {patient.age}"
    try:
        report1 = create_person_report("P001", template1)
        print("Report 1:")
        print(report1)
    except ValueError as e:
        print(f"Error generating report 1: {e}")

    print("-" * 20)

    # Example 2: Successful report generation with different template
    template2 = "Patient Name: {patient.name}, Gender: {patient.gender}"
    try:
        report2 = create_person_report("P001", template2)
        print("Report 2:")
        print(report2)
    except ValueError as e:
        print(f"Error generating report 2: {e}")

    print("-" * 20)

    # Example 3: Attempting to access sensitive information (should raise ValueError)
    template_sensitive = "Patient Name: {patient.name}\nDiagnosis: {patient.diagnosis}"
    try:
        report_sensitive = create_person_report("P001", template_sensitive)
        print("Sensitive Report (should not print):")
        print(report_sensitive)
    except ValueError as e:
        print(f"Error generating sensitive report: {e}")

    print("-" * 20)

    # Example 4: Patient ID not found (should raise ValueError)
    try:
        report_not_found = create_person_report("P002", template1)
        print("Non-existent Patient Report (should not print):")
        print(report_not_found)
    except ValueError as e:
        print(f"Error generating report for non-existent patient: {e}")

    print("-" * 20)
    
    # Example 5: Template with an unknown/unsupported placeholder (should raise ValueError)
    template_unknown = "Patient Name: {patient.name}\nFavorite Color: {patient.favorite_color}"
    try:
        report_unknown = create_person_report("P001", template_unknown)
        print("Unknown Placeholder Report (behavior depends on implementation):")
        print(report_unknown)
    except ValueError as e:
        print(f"Error generating report with unknown placeholder: {e}")
