from utils.constants import SEQUENCE_LENGTH


def get_unique_characters(data):
    return sorted(list(set(data)))


def create_character_mappings(characters):
    int_to_char_mapping = {integer: char for integer, char in enumerate(characters)}
    char_to_int_mapping = {char: integer for integer, char in enumerate(characters)}
    return (int_to_char_mapping, char_to_int_mapping)


def create_training_and_target_list(data, char_to_int_mapping):
    training_list = []
    target_list = []

    for i in range(0, len(data) - SEQUENCE_LENGTH):
        sequence = data[i : i + SEQUENCE_LENGTH]
        target = data[i + SEQUENCE_LENGTH]
        training_list.append([char_to_int_mapping[character] for character in sequence])
        target_list.append(char_to_int_mapping[target])

    return (training_list, target_list)
