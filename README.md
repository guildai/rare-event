# Rare Event Prediction

[These models](guild.yml) are adapted from these blog posts:

- [Extreme Rare Event Classification using Autoencoders in Keras](https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098)
- [LSTM Autoencoder for Extreme Rare Event Classification in Keras](https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb)

by [Chitta Ranjan](https://www.linkedin.com/in/chitta-ranjan-b0851911/)

The original source is included as Notebooks:

- [autoencoder_classifier.ipynb](autoencoder_classifier.ipynb)
- [lstm_autoencoder_classifier.ipynb](lstm_autoencoder_classifier.ipynb)

To train the models, use:

    $ guild run ae:train
    $ guild run lstm:train

The LSTM does not include validation accuracy.

## To Do

- [ ] Generate sample log (treat as simulation problem)
  - Contains mostly normal log events of whatever (negative example)
  - Supports SIGTERM or some other signal
  - Prints signal
  - After some period with a random component, logs a "crash"
    (positive example)
- [ ] Convert simulated logs into format we can train
- [x] Activation functions (elu, leaky relu, etc) (see advanced
      activations in Keras)
- [x] More or fewer layers
- [ ] Different optimizers
- [ ] Within the LSTM:
  - Dropout
  - ???
- [x] Bump epochs to 1000
- [x] Add early stopping (Keras callback)
- [ ] Learning rate schedules
- [ ] Use custom Keras metic for roc_auc (unless slows training)
- [ ] Check if metrics for LSTM is slowing training

### Bug in data processing

- Losing a column somehow
- He's using the row number in the xs, which masks the missing col

------------------

- Highlight feature engineering in data-preparation (convert from raw
  to prepared - time shift of y values)

- Use validation data for examples
