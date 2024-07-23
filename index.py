import tensorflow as tf # type: ignore
from tensorflow.keras.applications import InceptionV3 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, add # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet')
model = Model(base_model.input, base_model.layers[-2].output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_image(model, img_path):
    img = preprocess_image(img_path)
    feature_vector = model.predict(img)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector

# Example captions data
captions_data = {
    'image1.jpg': ['a cat sitting on a mat', 'a kitten resting on a rug'],
    'image2.jpg': ['a dog playing with a ball', 'a puppy in a park'],
}

# Preprocess captions
all_captions = []
for key, captions in captions_data.items():
    for cap in captions:
        all_captions.append(cap)

word_count = {}
for caption in all_captions:
    for word in caption.split(' '):
        word_count[word] = word_count.get(word, 0) + 1

vocab = [word for word in word_count if word_count[word] >= 1]
word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}
index_to_word = {idx + 1: word for idx, word in enumerate(vocab)}
vocab_size = len(word_to_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)

# Data generator
def data_generator(captions_data, model, word_to_index, max_length, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for img_path, captions in captions_data.items():
            feature_vector = encode_image(model, img_path)
            for caption in captions:
                seq = [word_to_index[word] for word in caption.split() if word in word_to_index]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature_vector)
                    X2.append(in_seq)
                    y.append(out_seq)
                if n == batch_size:
                    yield [[np.array(X1), np.array(X2)], np.array(y)]
                    X1, X2, y = [], [], []
                    n = 0
                n += 1

# Build the model
input_img_features = tf.keras.Input(shape=(2048,))
img_features = Dense(256, activation='relu')(input_img_features)

input_sequence = tf.keras.Input(shape=(max_length,))
sequence_features = Embedding(vocab_size, 256, mask_zero=True)(input_sequence)
sequence_features = LSTM(256)(sequence_features)

decoder = add([img_features, sequence_features])
decoder = Dense(256, activation='relu')(decoder)
output = Dense(vocab_size, activation='softmax')(decoder)

model = tf.keras.Model(inputs=[input_img_features, input_sequence], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
epochs = 10
batch_size = 32
steps = len(captions_data) // batch_size

for i in range(epochs):
    generator = data_generator(captions_data, model, word_to_index, max_length, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Generate captions
def generate_caption(model, img_path, word_to_index, index_to_word, max_length):
    feature_vector = encode_image(model, img_path)
    input_seq = [word_to_index.get('<start>', 0)]
    for _ in range(max_length):
        seq = pad_sequences([input_seq], maxlen=max_length)
        yhat = model.predict([np.array([feature_vector]), np.array(seq)], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word.get(yhat, '')
        input_seq.append(yhat)
        if word == '<end>':
            break
    return ' '.join([index_to_word[i] for i in input_seq if i in index_to_word])

# Example usage
img_path = 'path_to_your_image.jpg'
caption = generate_caption(model, img_path, word_to_index, index_to_word, max_length)
print("Caption:", caption)
