# imports
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing import sequence
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(3)
# %matplotlib inline

# load dataset
vocab_size = 5000
max_input_word_length = 300
word_vec_features = 32
batch_size = 64
epochs = 13

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
# to get word to id mapping based on the frequency of occurrence
#    word2id = imdb.get_word_index()
# to get id to word mapping
#    id2word = {i: word for word, i in word2id.items()}

print("{} training examples".format(len(x_train)))
print("{} testing examples".format(len(x_test)))

# add padding to make each training example to have same length (if more words by truncating them or if less words by adding zeros)
x_train = sequence.pad_sequences(x_train, maxlen=max_input_word_length, padding='pre', truncating='pre')
x_test = sequence.pad_sequences(x_test, maxlen=max_input_word_length, padding='pre', truncating='pre')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# model
model = Sequential()
model.add(Embedding(vocab_size, word_vec_features, input_length=max_input_word_length))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# evaluate the model on test data
score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

print("Test score =", score)
print("Test accuracy =", accuracy)

# plot model accuracy for train and test data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuray')
plt.xlabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot loss for train and test data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save model and weights
model_json = model.to_json()
with open("imdb_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("imdb_model_weights.h5")
print("Model and Weights saved.!")
