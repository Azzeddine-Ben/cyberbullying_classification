# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:06:34 2021

@author: IT Doctor
"""
import numpy as np
import random as python_random
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Activation

#### Bert Model
def mcnn_model_tpu(bert_model):
  np.random.seed(123)
  python_random.seed(123)
  tf.random.set_seed(1234)

  input_ids          = keras.layers.Input(shape=(40,),dtype='int32')
  attention_masks    = keras.layers.Input(shape=(40,),dtype='int32')
  use_inputs         = keras.layers.Input(shape=(512,),dtype='float32')

  output = bert_model([input_ids,attention_masks])
  output = output[0]
  # bilstm = keras.layers.Bidirectional(keras.layers.LSTM(40, activation='relu', return_sequences=True))(output)
  conv1 = keras.layers.Conv1D(32, 2, activation='relu')(output)
  conv2 = keras.layers.Conv1D(32, 4, activation='relu')(output)
  conv3 = keras.layers.Conv1D(32, 6, activation='relu')(output)

  pool1 = keras.layers.GlobalMaxPooling1D()(conv1)
  pool2 = keras.layers.GlobalMaxPooling1D()(conv2)
  pool3 = keras.layers.GlobalMaxPooling1D()(conv3)


  concat = keras.layers.Concatenate()([pool1, pool2, pool3, use_inputs])

  dense = keras.layers.Dense(64,activation='relu')(concat)
  dense = keras.layers.Dropout(0.2)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)

  # loss_ = SigmoidFocalCrossEntropy_1(alpha=0.65, gamma=0.55)

  model = keras.models.Model(inputs = [input_ids, attention_masks, use_inputs],outputs = output)
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model

def blstm_mcnn_model_tpu(bert_model):
  np.random.seed(123)
  python_random.seed(123)
  tf.random.set_seed(1234)

  input_ids          = keras.layers.Input(shape=(40,),dtype='int32')
  attention_masks    = keras.layers.Input(shape=(40,),dtype='int32')
  use_inputs         = keras.layers.Input(shape=(512,),dtype='float32')


  output = bert_model([input_ids,attention_masks])
  output = output[0]
  bilstm = keras.layers.Bidirectional(keras.layers.LSTM(40, activation='relu', return_sequences=True))(output)
  conv1 = keras.layers.Conv1D(32, 2, activation='relu')(bilstm)
  conv2 = keras.layers.Conv1D(32, 4, activation='relu')(bilstm)
  conv3 = keras.layers.Conv1D(32, 6, activation='relu')(bilstm)

  pool1 = keras.layers.GlobalMaxPooling1D()(conv1)
  pool2 = keras.layers.GlobalMaxPooling1D()(conv2)
  pool3 = keras.layers.GlobalMaxPooling1D()(conv3)

  concat = keras.layers.Concatenate()([pool1, pool2, pool3, use_inputs])

  dense = keras.layers.Dense(64,activation='relu')(concat)
  dense = keras.layers.Dropout(0.2)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)

  # loss_ = SigmoidFocalCrossEntropy_1(alpha=0.65, gamma=0.55)

  model = keras.models.Model(inputs = [input_ids, attention_masks, use_inputs], outputs = output)
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model


def mcnn_blstm_model_tpu(bert_model):
  np.random.seed(123)
  python_random.seed(123)
  tf.random.set_seed(1234)

  input_ids          = keras.layers.Input(shape=(40,),dtype='int32')
  attention_masks    = keras.layers.Input(shape=(40,),dtype='int32')
  use_inputs         = keras.layers.Input(shape=(512,),dtype='float32')
  
  output = bert_model([input_ids,attention_masks])
  output = output[0]
  
  conv1 = keras.layers.Conv1D(32, 2, activation='relu', padding='same')(output)
  conv2 = keras.layers.Conv1D(32, 4, activation='relu', padding='same')(output)
  conv3 = keras.layers.Conv1D(32, 6, activation='relu', padding='same')(output)

  pool1 = keras.layers.MaxPooling1D()(conv1)
  pool2 = keras.layers.MaxPooling1D()(conv2)
  pool3 = keras.layers.MaxPooling1D()(conv3)
  
  pool_concat = keras.layers.Concatenate()([pool1, pool2, pool3])
  
  bilstm = keras.layers.Bidirectional(keras.layers.LSTM(40, activation='relu'))(pool_concat)
  concat = keras.layers.Concatenate()([bilstm, use_inputs])

  dense = keras.layers.Dense(64,activation='relu')(concat)
  dense = keras.layers.Dropout(0.2)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)

  # loss_ = SigmoidFocalCrossEntropy_1(alpha=0.65, gamma=0.55)

  model = keras.models.Model(inputs = [input_ids, attention_masks, use_inputs], outputs = output)
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model

####################################################
######      Models with input features        ######
####################################################

def mcnn_model_tpu_with_features(bert_model):
  np.random.seed(123)
  python_random.seed(123)
  tf.random.set_seed(1234)

  input_ids          = keras.layers.Input(shape=(40,),dtype='int32')
  attention_masks    = keras.layers.Input(shape=(40,),dtype='int32')
  use_inputs         = keras.layers.Input(shape=(512,),dtype='float32')

  input_lexical     = keras.layers.Input(shape=(17,),dtype='float32')
  input_readability = keras.layers.Input(shape=(16,),dtype='float32')
  input_sentiments  = keras.layers.Input(shape=(6,),dtype='float32')
  input_stylometric = keras.layers.Input(shape=(4,),dtype='float32')
  
  output = bert_model([input_ids,attention_masks])
  output = output[0]
  # bilstm = keras.layers.Bidirectional(keras.layers.LSTM(40, activation='relu', return_sequences=True))(output)
  conv1 = keras.layers.Conv1D(32, 2, activation='relu')(output)
  conv2 = keras.layers.Conv1D(32, 4, activation='relu')(output)
  conv3 = keras.layers.Conv1D(32, 6, activation='relu')(output)

  pool1 = keras.layers.GlobalMaxPooling1D()(conv1)
  pool2 = keras.layers.GlobalMaxPooling1D()(conv2)
  pool3 = keras.layers.GlobalMaxPooling1D()(conv3)


  concat = keras.layers.Concatenate()([pool1, pool2, pool3, use_inputs, input_lexical, input_readability, input_sentiments, input_stylometric])

  dense = keras.layers.Dense(64,activation='relu')(concat)
  dense = keras.layers.Dropout(0.2)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)

  # loss_ = SigmoidFocalCrossEntropy_1(alpha=0.65, gamma=0.55)

  model = keras.models.Model(inputs = [input_ids, attention_masks, use_inputs, input_lexical, input_readability, input_sentiments, input_stylometric],outputs = output)
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model

def blstm_mcnn_model_tpu_with_features(bert_model):
  np.random.seed(123)
  python_random.seed(123)
  tf.random.set_seed(1234)

  input_ids          = keras.layers.Input(shape=(40,),dtype='int32')
  attention_masks    = keras.layers.Input(shape=(40,),dtype='int32')
  use_inputs         = keras.layers.Input(shape=(512,),dtype='float32')

  input_lexical     = keras.layers.Input(shape=(17,),dtype='float32')
  input_readability = keras.layers.Input(shape=(16,),dtype='float32')
  input_sentiments  = keras.layers.Input(shape=(6,),dtype='float32')
  input_stylometric = keras.layers.Input(shape=(4,),dtype='float32')

  output = bert_model([input_ids,attention_masks])
  output = output[0]
  bilstm = keras.layers.Bidirectional(keras.layers.LSTM(40, activation='relu', return_sequences=True))(output)
  conv1 = keras.layers.Conv1D(32, 2, activation='relu')(bilstm)
  conv2 = keras.layers.Conv1D(32, 4, activation='relu')(bilstm)
  conv3 = keras.layers.Conv1D(32, 6, activation='relu')(bilstm)

  pool1 = keras.layers.GlobalMaxPooling1D()(conv1)
  pool2 = keras.layers.GlobalMaxPooling1D()(conv2)
  pool3 = keras.layers.GlobalMaxPooling1D()(conv3)

  concat = keras.layers.Concatenate()([pool1, pool2, pool3, use_inputs, input_lexical, input_readability, input_sentiments, input_stylometric])

  dense = keras.layers.Dense(64,activation='relu')(concat)
  dense = keras.layers.Dropout(0.2)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)

  # loss_ = SigmoidFocalCrossEntropy_1(alpha=0.65, gamma=0.55)

  model = keras.models.Model(inputs = [input_ids, attention_masks, use_inputs, input_lexical, input_readability, input_sentiments, input_stylometric], outputs = output)
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model


def mcnn_blstm_model_tpu_with_features(bert_model):
  np.random.seed(123)
  python_random.seed(123)
  tf.random.set_seed(1234)

  input_ids          = keras.layers.Input(shape=(40,),dtype='int32')
  attention_masks    = keras.layers.Input(shape=(40,),dtype='int32')
  use_inputs         = keras.layers.Input(shape=(512,),dtype='float32')
  
  input_lexical     = keras.layers.Input(shape=(17,),dtype='float32')
  input_readability = keras.layers.Input(shape=(16,),dtype='float32')
  input_sentiments  = keras.layers.Input(shape=(6,),dtype='float32')
  input_stylometric = keras.layers.Input(shape=(4,),dtype='float32')

  output = bert_model([input_ids,attention_masks])
  output = output[0]
  
  conv1 = keras.layers.Conv1D(32, 2, activation='relu', padding='same')(output)
  conv2 = keras.layers.Conv1D(32, 4, activation='relu', padding='same')(output)
  conv3 = keras.layers.Conv1D(32, 6, activation='relu', padding='same')(output)

  pool1 = keras.layers.MaxPooling1D()(conv1)
  pool2 = keras.layers.MaxPooling1D()(conv2)
  pool3 = keras.layers.MaxPooling1D()(conv3)
  
  pool_concat = keras.layers.Concatenate()([pool1, pool2, pool3])
  
  bilstm = keras.layers.Bidirectional(keras.layers.LSTM(40, activation='relu'))(pool_concat)
  concat = keras.layers.Concatenate()([bilstm, use_inputs, input_lexical, input_readability, input_sentiments, input_stylometric])

  dense = keras.layers.Dense(64,activation='relu')(concat)
  dense = keras.layers.Dropout(0.2)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)

  # loss_ = SigmoidFocalCrossEntropy_1(alpha=0.65, gamma=0.55)

  model = keras.models.Model(inputs = [input_ids, attention_masks, use_inputs, input_lexical, input_readability, input_sentiments, input_stylometric], outputs = output)
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
  # model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model