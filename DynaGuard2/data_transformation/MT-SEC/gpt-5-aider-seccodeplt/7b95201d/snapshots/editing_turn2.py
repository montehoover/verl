import re

def analyze_text(text_content):
    if not text_content:
        return {'word_count': 0, 'word_lengths': []}
    words = re.findall(r"\b\w+\b", text_content)
    word_lengths = [len(word) for word in words]
    return {'word_count': len(words), 'word_lengths': word_lengths}
