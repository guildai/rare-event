# Credit: https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation
from keras.layers import LeakyReLU, PReLU, ELU, ThresholdedReLU

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2
DATA = 'data/rare-events-prepared.csv'
samples = 50

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

# Directories for generated content
dirs = ['logs', 'plots', 'checkpoints']
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)

df = pd.read_csv(DATA)
df_train, df_test = train_test_split(df, test_size=DATA_SPLIT_PCT, random_state=SEED)
df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

df_train_0 = df_train.loc[df['y'] == 0]
df_train_1 = df_train.loc[df['y'] == 1]
df_train_0_x = df_train_0.drop(['y'], axis=1)
df_train_1_x = df_train_1.drop(['y'], axis=1)

df_valid_0 = df_valid.loc[df['y'] == 0]
df_valid_1 = df_valid.loc[df['y'] == 1]
df_valid_0_x = df_valid_0.drop(['y'], axis=1)
df_valid_1_x = df_valid_1.drop(['y'], axis=1)

df_test_0 = df_test.loc[df['y'] == 0]
df_test_1 = df_test.loc[df['y'] == 1]
df_test_0_x = df_test_0.drop(['y'], axis=1)
df_test_1_x = df_test_1.drop(['y'], axis=1)

scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['y'], axis = 1))

with open("x-samples.json", "w") as f:
    json.dump(df_valid_x_rescaled[:samples].tolist(), f)

df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['y'], axis = 1))

nb_epoch = 200
batch_size = 128
input_dim = df_train_0_x_rescaled.shape[1] #num of predictor variables,
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-3
activation = "relu"
patience = 5
encoder_hidden_layers = 1
decoder_hidden_layers = 1
optimizer = "adam"

def DefaultActivation():
    if activation == "leaky-relu":
        return LeakyReLU()
    elif activation == "prelu":
        return PReLU()
    elif activation == "threshold-relu":
        return ThresholdedReLU()
    else:
        return Activation(activation)

def Hidden(n):
    def f(model):
        for _ in range(n):
            model = Dense(hidden_dim)(model)
            model = DefaultActivation()(model)
        return model
    return f

input_layer = Input(shape=(input_dim, ))

# Encoder layers
encoder = Dense(encoding_dim, activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = DefaultActivation()(encoder)
encoder = Hidden(encoder_hidden_layers)(encoder)

# Decoder layers
decoder = Hidden(decoder_hidden_layers)(encoder)
decoder = Dense(encoding_dim)(decoder)
decoder = DefaultActivation()(decoder)
decoder = Dense(input_dim, activation="linear")(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

with open('checkpoints/model.json', 'w') as f:
    f.write(autoencoder.to_json())

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=optimizer)

cp = ModelCheckpoint("checkpoints/weights.h5", save_best_only=True)
tb = TensorBoard()
es = EarlyStopping(patience=patience)

history = autoencoder.fit(
    df_train_0_x_rescaled, df_train_0_x_rescaled,
    epochs=nb_epoch,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
    verbose=1,
    callbacks=[cp, tb, es]).history

autoencoder = load_model('checkpoints/weights.h5')

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('plots/loss.png')

valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_valid['y']})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.savefig('plots/precision-recall.png')

test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_test['y']})
error_df_test = error_df_test.reset_index()

threshold_fixed = 0.4
groups = error_df_test.groupby('True_class')

fig, ax = plt.subplots()

""" TEMP DISABLED due to traceback:
Traceback (most recent call last):
  File "/home/garrett/SCM/appian-hc/models/rare/train_ae.py", line 201, in <module>
    label= "Break" if name == 1 else "Normal")
  File "/home/garrett/SCM/appian-hc/models/rare/env/lib/python3.6/site-packages/matplotlib/__init__.py", line 1810, in inner
    return func(ax, *args, **kwargs)
  File "/home/garrett/SCM/appian-hc/models/rare/env/lib/python3.6/site-packages/matplotlib/axes/_axes.py", line 1611, in plot
    for line in self._get_lines(*args, **kwargs):
  File "/home/garrett/SCM/appian-hc/models/rare/env/lib/python3.6/site-packages/matplotlib/axes/_base.py", line 393, in _grab_next_args
    yield from self._plot_args(this, kwargs)
  File "/home/garrett/SCM/appian-hc/models/rare/env/lib/python3.6/site-packages/matplotlib/axes/_base.py", line 378, in _plot_args
    ncx, ncy = x.shape[1], y.shape[1]
IndexError: tuple index out of range

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.savefig("plots/reconstruction-error.png")
"""

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

predictions = pd.DataFrame({'true': error_df.True_class,
                           'predicted': pred_y})

conf_matrix = confusion_matrix(error_df.True_class, pred_y)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig('plots/confusion-matrix.png')

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('plots/roc-curve.png')

print("roc_auc: %s" % roc_auc)
