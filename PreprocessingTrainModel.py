import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping# type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

from Config import DATA_PATH, actions, sequence_length


#PRE-PROCESSING
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Building and training the LSTM neural network

log_dir = os.path.join('Logs1')
tb_callback = TensorBoard(log_dir=log_dir)
'''
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu',kernel_regularizer=l2(0.01)))
model.add(Dense(32, activation='relu',kernel_regularizer=l2(0.01)))
model.add(Dense(actions.shape[0], activation='softmax'))
'''
model = Sequential()
model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(30, 1662)))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.15, epochs=300, callbacks=[early_stopping,tb_callback])
model.summary()


'''
OPEN TESNSORBOARD:
    1.Open Terminal
    2.Go to the folder
    3.cd Logs
    4.cd train
    5.tensorboard --logdir=.
    6.It will give link localhost
    7.copy past in browser
    8.check the logs for accuracy,loss,architecture,time series data
    9.check for drops to find number of epochs

'''

model.save('action.h5')


ytest = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
ytest = np.argmax(ytest, axis=1).tolist()
print("Testing Accuracy: ",accuracy_score(ytrue, ytest))
