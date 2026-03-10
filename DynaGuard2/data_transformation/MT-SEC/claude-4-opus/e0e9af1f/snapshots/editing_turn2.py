def read_text_file(message):
    with open('log.txt', 'a') as file:
        file.write(message + '\n')
