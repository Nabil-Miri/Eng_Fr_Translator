import os

# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from PIL import Image
from urllib.request import urlretrieve

import logging
import pickle
import numpy as np


class EnFrTranslator:

    def __init__(self, model_path):
        logging.info("EnFrTranslator class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")

    def get_tokenizers(self):
        # loading
        print('BEFORE')
        with open("Tokenizers/tokenizer_eng.pickle", 'rb') as handle:
            print('AFTER')
            self.tokenizer_eng = pickle.load(handle)

        with open('Tokenizers/tokenizer_fr.pickle', 'rb') as handle:
            self.tokenizer_fr = pickle.load(handle)

    # Test Your Zaka
    def translate(self, eng_sentence):
        eng_max_length = 15
        fr_max_length = 21
        self.get_tokenizers()
        sentence_token = self.tokenizer_eng.texts_to_sequences([eng_sentence])
        sentence_pad = pad_sequences(sentence_token, maxlen=eng_max_length, padding='post')

        prediction = self.model.predict(sentence_pad)

        translated_words = np.argmax(prediction,
                                     axis=2)  # argmax across axis 2 gives us (21, 345) where we have 21 max length sentence and 345 possible word
        no_zero = [i for i in translated_words[0] if i != 0]  # removing zeros

        translated_sentence = [list(self.tokenizer_fr.word_index)[word - 1] for word in no_zero]
        translated_sentence = ' '.join(translated_sentence)

        print(f'Eng Sentence:{eng_sentence}')
        print(f'Fr Sentence:{translated_sentence}')
        return translated_sentence

def main():

    model = EnFrTranslator('2BiLSTM.h5')
    predicted_class = model.translate("She is driving the truck")
    logging.info("This is an image of a {}".format(predicted_class))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
