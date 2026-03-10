def read_text_file(message: str) -> None:
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")
