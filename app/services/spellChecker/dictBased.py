from nltk.metrics.distance import edit_distance
from numpy import dot
from numpy.linalg import norm


def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = set(file.read().splitlines())
    return words


hebrew_words = load_words('/Users/yovel.c/PycharmProjects/services/sublineStreamlit/samples/hebWords.txt')


def cosine_similarity(vec_a, vec_b):
    return dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def find_closest_embedding(word, model, similarity_threshold=0.94):
    if word in hebrew_words:
        return word  # Word is correct

    if word not in model:
        return word  # Word not in embeddings, return as is

    word_vector = model[word]
    closest_word = word
    highest_similarity = 0

    for w in hebrew_words:
        if w in model:
            similarity = cosine_similarity(word_vector, model[w])
            if similarity > highest_similarity and similarity >= similarity_threshold:
                highest_similarity = similarity
                closest_word = w

    return closest_word if highest_similarity >= similarity_threshold else word


def process_sentence(sentence, word_list, model):
    words = sentence.split()
    corrected_words = [find_closest_embedding(word, word_list, model) for word in words]
    return ' '.join(corrected_words)


if __name__ == '__main__':
    s = ''
    p = process_sentence()
