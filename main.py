import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hides TensorFlow warnings

start_time = time.time()
img_width, img_height = 100, 100

train_data_dir = os.getcwd() + '/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
test_data_dir = os.getcwd() + '/asl-alphabet/asl_alphabet_test/'

if os.path.isfile('./model/trained_model.h5') and os.path.isfile('./model/trained_history.npy'):
    model = load_model('./model/trained_model.h5')
    trained_history = np.load('./model/trained_history.npy', allow_pickle=True).item()
    model.name = 'ASL Model'
    model.summary()
    plot_model(model, show_shapes=True, expand_nested=True, to_file='./model/model.png')
    ann_viz(model)
else:
    epochs = 50
    batch_size = 290
    input_shape = (img_width, img_height, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(29, activation='softmax'))

    model.name = 'ASL Model'
    model.summary()
    plot_model(model, show_shapes=True, expand_nested=True, to_file='./model/model.png')
    ann_viz(model)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        subset='validation')

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

    es_callback = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        callbacks=[es_callback]
    )

    trained_history = history.history
    np.save('./model/model_history', trained_history)
    model.save('./model/model.h5')
    model.evaluate_generator(validation_generator, STEP_SIZE_VALID, verbose=1)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode='grayscale',
    class_mode=None,
    shuffle=False)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()

prediction = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)

predicted_class_indices = np.argmax(prediction, axis=1)

labels = [f.name for f in os.scandir(train_data_dir) if f.is_dir()]
labels = sorted(labels)
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv('./model/model_results.csv', index=False)

hours, rem = divmod(time.time() - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Execution Time:", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

plot1 = plt.figure(1)
plt.plot(trained_history['accuracy'])
plt.plot(trained_history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./model/model_accuracy.png')

plot2 = plt.figure(2)
plt.plot(trained_history['loss'])
plt.plot(trained_history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./model/model_loss.png')
plt.show()
