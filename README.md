# Deep learning text generator

A python script that uses deep learning to generate text - inspired by [this blog post](https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/).

## How it works
This script uses a Long short-term memory (LSTM) model and the parameters can be tweaked in `/utils/constants.py`.

### Input file
You can provide a text file that will serve as the input for the algorithm to start learning.
1. Make sure to save your input file in `/input`
2. Update the filename of `INPUT_FILE` in `/utils/constants.py`.

### Weights
Update the directory of `WEIGHTS_FILE` in `/utils/constants.py`.
- If the weights file already exists, weights will be loaded from file when generating text.
- If the weights file does not yet exist, weights will be generated and saved to the provided file name before generating text.

## Running the script

First install packages by running
```bash
$ ./build
```

To start generating text, run
```bash
$ source ./env/bin/activate
$ python generate_text.py
```

To deactivate the virtualenv when done, run
```bash
$ deactivate
```

## Feeling festive?
This repo contains an input file and some weights for generating Christmas Greetings. Very handy for writing some funny AI generated Christmas cards.

Make sure to set the following values in `/utils/constants.py` and you're good to go.

```bash
INPUT_FILE = "./input/christmas_greetings.txt"
WEIGHTS_FILE = "./weights/greetings_generator_50epochs.h5"
SEQUENCE_LENGTH = 50
```

Here's an example of Christmas Greetings the model can output.

```bash
Happy holidays.

Whishing you every happiness this christmas bnd throughout the new year ahead.

Have an ideal christmas an occasion that io celebrated as a reflection of your values, desires, affections, traditions.
```

## Special thanks
Special thanks to Codebar Brighton for organising a [Christmas show and tell](https://www.codebar.io/events/xmas-2020) which motivated me to finally have a play with machine learning.
