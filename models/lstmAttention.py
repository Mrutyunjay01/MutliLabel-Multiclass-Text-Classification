import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Dropout, Dense, Input, Embedding, Bidirectional, LSTM, Concatenate


class AttentionBlock(keras.Model):

    def __init__(self, units):
        super(AttentionBlock, self).__init__()
        self.W1 = Dense(units=units)
        self.W2 = Dense(units=units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_t = K.expand_dims(hidden, 1)
        # additive attention
        score = K.tanh(self.W1(features) + self.W2(hidden_t))

        attn_weights = K.softmax(self.V(score), axis=1)

        context = attn_weights * features
        context = tf.reduce_sum(context, axis=1)

        return context, attn_weights
        pass

    pass


def build_model(max_len, max_features, embed_size, attn_units=20, num_classes=4, rnn_cell_size=32):
    seq_inp = Input(shape=max_len, dtype="int32")
    embedded_seq = Embedding(max_features, embed_size)(seq_inp)
    lstm = Bidirectional(LSTM(
        rnn_cell_size, return_sequences=True
    ), name="bilstm_0")(embedded_seq)

    lstm, f_h, f_c, b_h, b_c = Bidirectional(LSTM(
        rnn_cell_size, return_sequences=True, return_state=True
    ), name="bilstm_1")(lstm)

    h_ = Concatenate()([f_h, b_h])
    c_ = Concatenate()([f_c, b_c])

    context, attn_weights = AttentionBlock(attn_units)(lstm, h_)

    fc_pre = Dense(num_classes * 4, activation="relu")(context)
    do = Dropout(0.05)(fc_pre)
    output = Dense(num_classes, activation="softmax")(do)

    return keras.Model(inputs=seq_inp, outputs=output)
    pass


if __name__ == '__main__':
    model = build_model(20, 100, 10000, 128)
    print(model.summary())
    pass
