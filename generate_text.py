#!/usr/bin/env python3
from utils.modelling import get_trained_model
from utils.output_text import generate_and_print_text
from utils.process_data import (
    create_character_mappings,
    create_training_and_target_list,
    get_unique_characters,
)
from utils.setup import disable_tensorflow_info_logs, load_data

# Setup
disable_tensorflow_info_logs()
data = load_data()

# Process data
characters = get_unique_characters(data)
(int_to_char_mapping, char_to_int_mapping) = create_character_mappings(characters)
(training_list, target_list) = create_training_and_target_list(
    data, char_to_int_mapping
)

# Modelling
model = get_trained_model(training_list, target_list, characters)

# Output text
generate_and_print_text(model, training_list, int_to_char_mapping, characters)
