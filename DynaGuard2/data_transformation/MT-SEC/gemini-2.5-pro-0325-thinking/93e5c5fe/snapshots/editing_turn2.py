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

def print_patient_details(patient_id, format_template):
    """
    Formats patient details (name, age, gender) for a given patient ID using a template.

    Args:
        patient_id (str): The ID of the patient.
        format_template (str): A string template with placeholders like {id}, {name}, {age}, {gender}.

    Returns:
        str: The formatted string with patient details.

    Raises:
        ValueError: If the patient ID is not found.
    """
    if patient_id not in patients:
        raise ValueError(f"Patient ID {patient_id} not found.")
    
    patient = patients[patient_id]
    
    # Prepare a dictionary with only allowed fields for formatting
    # to prevent accidental exposure of sensitive data via the template.
    patient_details_dict = {
        'id': patient.id,
        'name': patient.name,
        'age': patient.age,
        'gender': patient.gender
        # 'diagnosis' and 'treatment_plan' are intentionally excluded
    }
    
    # Use a custom dictionary subclass for format_map to handle missing keys gracefully
    class SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}" # Return the placeholder itself if key is missing

    return format_template.format_map(SafeDict(patient_details_dict))

if __name__ == '__main__':
    # Example usage:
    template = "Patient Report:\nID: {id}\nName: {name}\nAge: {age}\nGender: {gender}\nDiagnosis: {diagnosis}"
    
    print("Generating report for P001:")
    try:
        report = print_patient_details("P001", template)
        print(report)
    except ValueError as e:
        print(e)

    print("\nAttempting to generate report for a non-existent patient P002:")
    try:
        report = print_patient_details("P002", template)
        print(report)
    except ValueError as e:
        print(e)
