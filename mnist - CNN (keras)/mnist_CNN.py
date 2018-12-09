from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras import backend as K

batch_size = 128
epochs = 12
num_classes = 10

img_rows = 28
img_columns = 28

(x_train,y_train),(x_test, y_test) = mnist.load_data()

if(K.image_data_format() == 'channels_first'):
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_columns)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_columns)
	input_shape = (1, img_rows, img_columns)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_columns, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_columns, 1)
	input_shape = (img_rows, img_columns, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Convolution2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))
model.add(Convolution2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])