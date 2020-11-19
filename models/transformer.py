# Create a transformer architecture like "Attention is all you need" style with NO LSTM/RNN/GRU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    """
    Implement the core concept of transformer which is this MultiHead Attention implemented as a layer.
    """

    def __init__(self, embed_dim=128, num_heads=8):
        """
        embed_dim: {int} Size of the Embedding Dimension 128 default
        num_heads: {int} Number of heads to be used. Default 8
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    """
    Implement a Transformer block
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    """
    Implementation of specialised Embedding based on Cosine Function
    """

    def __init__(self, maxlen, vocab_size, embed_dim):
        """
        maxlen: {int} maximum length to use for sentence. if 100, use first 100 tokens only per sentence
        vocab_size: {int} size of the vocablury
        embed_dim: {int} size of the embedding dimension
        """
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def build_transformer_model(n_classes=4, activation='softmax', hidden_dim=32, maxlen=254, vocab_size=20000,
                            embed_dim=128, num_heads=8, dropout_fact=0.2, dense_size=32):
    """
    Build the model
    args:
        n_classes: {int} number of classes to classify in the last layer
        activation: {str/keras.activations} activation function to use for the last layer
        hidden_dim: {int} neurons to use in the Hidden Dense Layer inside Transformer
        maxlen: {int} macximum length for the sentennce
        vocab_size: {int} size of the vocablary
        embed_dim: {int} size of the Embedded Dimension for each token. Each token will be represented by this new Dim
        num_heads: {int} Number of heads to use in the Transformer
        dropout_fact: {float} dropout factor to use affter Dense Layers
        dense_size: {int} Size of the Last Dense Layer (before output)

    return:
        instance of keras.model with given configurations
    """

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, hidden_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_fact)(x)
    x = layers.Dense(dense_size, activation="relu")(x)
    x = layers.Dropout(dropout_fact)(x)
    outputs = layers.Dense(n_classes, activation=activation)(x)

    return keras.Model(inputs=inputs, outputs=outputs)
