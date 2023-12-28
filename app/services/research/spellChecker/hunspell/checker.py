import subprocess

_dict_path = '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/app/services/spellChecker/hunspell/index.dic'
_cmd = ['hunspell', '-d', _dict_path, '-a', '-i', 'utf-8']


def check_spelling(word):
    # Run the command and pass the word to be checked
    result = subprocess.run(_cmd, input=word, text=True, capture_output=True, encoding='utf-8')

    # Hunspell outputs each line starting with '*' (correct), '&' (misspelled), or '#' (unknown)
    return result.stdout


# Example usage

if __name__ == '__main__':
    sentence = ''
    output = check_spelling(sentence)
    print(output)
