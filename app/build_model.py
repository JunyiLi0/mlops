import pandas as pd 
import joblib
import numpy as np
import skimage
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

def build_model():

    IMAGE_PATH = '/mnt/c/Users/ASUS/OneDrive/Bureau/Cours/MLOPS/chinese mnist/data/data/'
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    IMAGE_CHANNELS = 1
    RANDOM_STATE = 2022
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    CONV_2D_DIM_1 = 16
    CONV_2D_DIM_2 = 16
    CONV_2D_DIM_3 = 32
    CONV_2D_DIM_4 = 64
    MAX_POOL_DIM = 2
    KERNEL_SIZE = 3
    BATCH_SIZE = 64
    NO_EPOCHS = 30
    DROPOUT_RATIO = 0.4
    PATIENCE = 5
    VERBOSE = 1

    data_df=pd.read_csv('/mnt/c/Users/ASUS/OneDrive/Bureau/Cours/MLOPS/chinese mnist/chinese_mnist.csv')

    def create_file_name(x):
        file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"
        return file_name
    data_df["file"] = data_df.apply(create_file_name, axis=1)

    def read_image_sizes(file_name):
        image = skimage.io.imread(IMAGE_PATH + file_name)
        return list(image.shape)

    m = np.stack(data_df['file'].apply(read_image_sizes))
    df = pd.DataFrame(m,columns=['w','h'])
    data_df = pd.concat([data_df,df],axis=1, sort=False)

    def read_image(file_name):
        image = skimage.io.imread(IMAGE_PATH + file_name)
        image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT, 1), mode='reflect')
        return image[:,:,:]

    def categories_encoder(dataset, var='character'):
        X = np.stack(dataset['file'].apply(read_image))
        y = pd.get_dummies(dataset[var], drop_first=False)
        return X, y

    train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data_df["code"].values)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_df["code"].values)

    def unique_in_order(lst):
        unique_lst = []
        for num in lst:
            if num not in unique_lst:
                unique_lst.append(num)
            if len(unique_lst) == 15:
                break
        return unique_lst

    X_train, y_train = categories_encoder(train_df)
    X_val, y_val = categories_encoder(val_df)
    X_test, y_test = categories_encoder(test_df)

    order = unique_in_order(y_train)
    joblib.dump(order, "label_order.joblib")

    model=Sequential()
    model.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))
    model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
    model.add(MaxPool2D(MAX_POOL_DIM))
    model.add(Dropout(DROPOUT_RATIO))
    model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
    model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
    model.add(Dropout(DROPOUT_RATIO))
    model.add(Flatten())
    model.add(Dense(y_train.columns.size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+NO_EPOCHS))
    earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)
    checkpointer = ModelCheckpoint('test.h5',
                                    monitor='val_accuracy',
                                    verbose=VERBOSE,
                                    save_best_only=True,
                                    save_weights_only=True)

    train_model  = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val),
                  callbacks=[earlystopper, checkpointer, annealer])

    model_optimal = model
    model_optimal.load_weights('best_model.h5')
    score = model_optimal.evaluate(X_test, y_test, verbose=0)
    print(f'Best validation loss: {score[0]}, accuracy: {score[1]}')

    joblib.dump(model_optimal, "best_model.joblib")