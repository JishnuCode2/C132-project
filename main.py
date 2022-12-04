import tensorflow
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ['good product']

tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)

word_index = tokenizer.word_index
seq = tokenizer.texts_to_sequences(sentence)

padded = pad_sequences(seq,maxlen=100,padding='post',truncating='post')

model = tensorflow.keras.models.load_model('C132-Project.h5')
result = model.predict(padded)

predict_class = np.argmax(result,axis=1)
print(predict_class)