import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

le = LabelEncoder()

import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from models.lstmAttention import build_model
from models.custom_full_attention_layer_with_bidirectional_lstm import build_ABDLstm_model
from models.transformer import build_transformer_model
from losses.loss_functions import FocalLoss

MAXLEN = 254
VOCAB_SIZE = 20000
EMBED_DIM = 128
BATCH_SIZE = 128
EPOCHS = 5
CLASSES = 198

TEXT_COL = "text"
TARGET_COL = "target"

apply_class_weights = True

if __name__ == '__main__':

    # load the preprocessed df directly
    df = pd.read_csv("./working_chapters.csv")
    df.dropna(inplace=True)
    df.reset_index(drop=True)
    assert len(df.columns) == 2, "Dataframe with text and target col only is allowed, check you df"
    df.columns = [TEXT_COL, TARGET_COL]
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    print("\nLength of train: ", len(train))
    print("Length of test: ", len(test))
    # Use train to crate further validation set
    # tokenize the sentences
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$&(),.:;?@[\\]^_`{|}\t\n')
    tokenizer.fit_on_texts(list(train[TEXT_COL]) + list(test[TEXT_COL]))
    word_idx = tokenizer.word_index

    print("====== Tokenizing texts ========")
    X_train = tokenizer.texts_to_sequences(list(train[TEXT_COL]))
    y_train = le.fit_transform(train[TARGET_COL].values)

    X_test = tokenizer.texts_to_sequences(list(test[TEXT_COL]))
    y_test = le.fit_transform(test[TARGET_COL])

    X_train = pad_sequences(X_train, maxlen=MAXLEN)
    X_test = pad_sequences(X_test, maxlen=MAXLEN)

    # apply cross validation
    print("===== Creating cross validation =========")
    cv_splits = list(StratifiedKFold(n_splits=5).split(X_train, y_train))

    oof_preds = np.zeros((X_train.shape[0], CLASSES))
    test_preds = np.zeros((X_test.shape[0], CLASSES))

    print("====== Build and compile the model ========")
    # model = build_model(max_len=MAXLEN, max_features=MAXWORDS, embed_size=EMBED_SIZE, num_classes=CLASSES)
    # model = build_transformer_model(n_classes=CLASSES)
    model = build_ABDLstm_model(n_classes=CLASSES, max_len=MAXLEN, vocab_size=VOCAB_SIZE)
    model.compile(loss=FocalLoss,
                  optimizer="adam",
                  metrics=["accuracy"])

    if apply_class_weights:
        print("Applying class weights...")
        cws = class_weight.compute_class_weight("balanced",
                                                classes=np.unique(y_train),
                                                y=y_train)
        cws = dict(enumerate(cws))

    print("Start training...")
    for fold in range(5):
        K.clear_session()
        train_idx, val_idx = cv_splits[fold]
        model.fit(X_train[train_idx], to_categorical(y_train[train_idx]),
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(X_train[val_idx], to_categorical(y_train[val_idx])),
                  class_weight=cws)

        oof_preds[val_idx] += model.predict(X_train[val_idx])
        test_preds += model.predict(X_test)
        pass
    test_preds /= 5

    print("Training Finished...")
    print("Performance in training Data...")
    oof_pred = K.argmax(oof_preds).numpy()
    print("ROC-AUC Score for training: ", roc_auc_score(y_true=y_train, y_score=oof_pred))
    print()
    print("Classification report for training: ", classification_report(y_true=y_train, y_pred=oof_pred))

    print("Evaluation on test data...")
    y_pred = K.argmax(test_preds).numpy()
    print("Classification report for training: ", classification_report(y_true=y_test, y_pred=y_pred))
    print("ROC-AUC Score for testing: ", roc_auc_score(y_true=y_test, y_score=y_pred))
    pass
