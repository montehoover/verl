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

def print_patient_details(patient_id):
    """
    Prints basic details (name, age, gender) for a given patient ID.

    Args:
        patient_id (str): The ID of the patient.

    Raises:
        ValueError: If the patient ID is not found.
    """
    if patient_id not in patients:
        raise ValueError(f"Patient ID {patient_id} not found.")
    
    patient = patients[patient_id]
    
    # Check for attempts to access sensitive information implicitly by function scope.
    # For this function, we are only accessing name, age, gender as requested.
    # If 'diagnosis' or 'treatment_plan' were requested by this function,
    # it would be considered an attempt to access sensitive information.
    # This check is more conceptual for this specific function, as it's designed
    # to only print non-sensitive fields.

    print(f"Patient ID: {patient.id}")
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Gender: {patient.gender}")

if __name__ == '__main__':
    # Example usage:
    try:
        print_patient_details("P001")
    except ValueError as e:
        print(e)

    print("\nAttempting to access a non-existent patient:")
    try:
        print_patient_details("P002")
    except ValueError as e:
        print(e)
