import keras.backend as K
from keras.layers import Layer
from keras.losses import categorical_crossentropy


def FocalLoss(inputs, targets, alpha=1, gamma=2):
    ce_loss = categorical_crossentropy(inputs, targets)
    pt = K.exp(-ce_loss)
    _focal_loss = alpha * (1 - pt)**gamma * ce_loss

    return _focal_loss
    pass


if __name__ == '__main__':
    pass
