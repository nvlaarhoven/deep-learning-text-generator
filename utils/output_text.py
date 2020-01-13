import numpy as np
import random

from utils.constants import (
    NUMBER_OF_CHARACTERS_TO_GENERATE,
    SEQUENCE_LENGTH,
)
from utils.common import reshape_and_normalize


def generate_text(model, training_list, int_to_char_mapping, characters):
    sequence = random.choice(training_list)
    text = [int_to_char_mapping[value] for value in sequence]

    for i in range(NUMBER_OF_CHARACTERS_TO_GENERATE):
        sequence_ndarray = reshape_and_normalize(
            sequence, (1, SEQUENCE_LENGTH, 1), len(characters)
        )

        next_int = np.argmax(model.predict(sequence_ndarray, verbose=0))
        text.append(int_to_char_mapping[next_int])

        sequence.append(next_int)
        sequence = sequence[1:]

    return "".join(text)


def cleanup_text(text):
    sentenses = text.replace(". ", ".\n").replace("! ", "!\n").split("\n")
    complete_sentenses = sentenses[1:-1] if len(sentenses) > 1 else sentenses
    capitalized_complete_sentenses = [
        sentense.capitalize() for sentense in complete_sentenses
    ]
    return "\n\n".join(capitalized_complete_sentenses)


def generate_and_print_text(model, training_list, int_to_char_mapping, characters):
    text = generate_text(model, training_list, int_to_char_mapping, characters)
    print(cleanup_text(text))
