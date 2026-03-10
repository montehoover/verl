def count_words(text):
    if not text or text.strip() == "":
        return 0
    return len(text.split())

def extract_words(text):
    if not text or text.strip() == "":
        return []
    return text.split()
