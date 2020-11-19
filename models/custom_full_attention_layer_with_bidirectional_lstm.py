import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Layer, Embedding, LSTM, Dense, Input


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(Attention, self).get_config()


def build_ABDLstm_model(n_classes, max_len, vocab_size):
    inputs = Input((max_len,))
    x = Embedding(input_dim=vocab_size + 1, output_dim=32, input_length=max_len,
                  embeddings_regularizer=keras.regularizers.l2(.001))(inputs)
    att_in = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)
    att_out = Attention()(att_in)
    outputs = Dense(n_classes, activation='sigmoid', trainable=True)(att_out)
    model = Model(inputs, outputs)
    return model
