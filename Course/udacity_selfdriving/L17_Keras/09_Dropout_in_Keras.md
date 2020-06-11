## Dropout

1. Build from the previous network.
2. Add a [dropout](https://keras.io/layers/core/#dropout) layer after the pooling layer. Set the dropout rate to 50%.
3. Make sure to note from the documentation above that the rate specified for dropout in Keras is the opposite of TensorFlow! TensorFlow uses the probability to *keep* nodes, while Keras uses the probability to *drop* them.

[Quiz](./Quiz_09_Dropout_in_Keras/09_Dropout_in_Keras.md)
