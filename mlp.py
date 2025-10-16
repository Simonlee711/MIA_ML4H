import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import os
import random




def train_test_split(df, train_percent, SEED):
  # train test split so that same subject id is not in both test and train

  unique_ids = df['subject_id'].unique()

  np.random.seed(SEED)
  np.random.shuffle(unique_ids)
  train_size = int(len(unique_ids) * train_percent)
  train_ids = unique_ids[:train_size]
  test_ids = unique_ids[train_size:]

  train_df = df[df['subject_id'].isin(train_ids)]
  test_df = df[df['subject_id'].isin(test_ids)]

  return train_df, test_df


def X_y_split(df, drop_columns, target_col):
  X = df.drop(columns=drop_columns)
  y = df[target_col]

  return X, y


def build_mlp_model(input_dim, task_type='classification', output_dim=None, hidden_dim=256, dropout_rate=0.3):

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    model.add(layers.Dense(hidden_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(hidden_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout_rate))

    if task_type == 'classification':
        model.add(layers.Dense(output_dim, activation='softmax'))
    else:  # regression
        model.add(layers.Dense(1))  

    return model





def compile_model(model, task_type='classification', weight_decay=1e-4):
    if task_type == 'classification':
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=weight_decay),
            loss='sparse_categorical_crossentropy',  
            metrics=['accuracy']
        )
    else:  # regression
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=weight_decay),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
    return model



def get_callbacks(use_early_stopping=True):
    callbacks = []
    if use_early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stop)
    return callbacks



def train_model(model, X_train, y_train, X_val, y_val,batch_size=128,epochs=50, use_early_stopping=True):

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=get_callbacks(use_early_stopping)
    )
    return history



def evaluate_model(model, X_test, y_test, task_type='classification'):
    results = model.evaluate(X_test, y_test)
    if task_type == 'classification':
        print(f"Test Accuracy: {results[1]:.4f}")
    else: # regression
        print(f"Test MAE: {results[1]:.4f}, MSE: {results[2]:.4f}")
    return results



def classification_metrics(model, X_test, y_test):
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    n_classes = y_probs.shape[1]  
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.plot([0, 0], [0, 1], 'k--')
    plt.plot([0, 1], [1, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)       
    random.seed(seed)                               
    np.random.seed(seed)                             
    tf.random.set_seed(seed)        
    


def attack_evaluation_set(train_df, val_df, test_df):
  #500 from train
  #500 from test and val combined
  val_test_df = pd.concat([val_df, test_df])

  member_df = train_df.sample(500, random_state = 42)
  non_member_df = val_test_df.sample(500, random_state = 42)

  return member_df, non_member_df



def negative_loss(model, x, y ):
  y_pred = model(x)

  loss = keras.losses.sparse_categorical_crossentropy(y, y_pred)

  negative_loss = -loss
  return negative_loss



def confidence_score(model, x, y):
    y_pred = model(x)  

    confidence_score = tf.gather(y_pred, y, axis=1, batch_dims=1)

    return confidence_score

                     

def seed_loop(target_df, target_col, drop_columns, K):

  """

  only one seed being tested right now (not actually looping through multiple seeds)
  need to change return so that it returns outputs for multiple models


  """

  seeds = [42]
  for seed in seeds:

      set_seed(seed)


      #outside loop?
      train_df, val_df = train_test_split(target_df, 0.6, SEED = seed)
      val_df, test_df = train_test_split(val_df, 0.5, SEED = seed)

      X_train, y_train = X_y_split(train_df, drop_columns, target_col)
      X_val, y_val = X_y_split(val_df, drop_columns, target_col)
      X_test, y_test = X_y_split(test_df, drop_columns, target_col)

      member_df, non_member_df = attack_evaluation_set(train_df, val_df, test_df)
      X_member, y_member = X_y_split(member_df, drop_columns, target_col)
      X_non_member, y_non_member = X_y_split(non_member_df, drop_columns, target_col)


      model = build_mlp_model(input_dim=X_train.shape[1],
                              task_type='classification',
                              output_dim=K,
                              hidden_dim=256,
                              dropout_rate=0.3)
      
      
      model = compile_model(model, task_type='classification')     
      
      
      train_model(model, X_train, y_train, X_val, y_val, use_early_stopping=True)
      evaluate_model(model, X_test, y_test, task_type='classification')
      classification_metrics(model, X_test, y_test)


      negative_loss_member = negative_loss(model, X_member, y_member)
      negative_loss_non_member = negative_loss(model, X_non_member, y_non_member)

      confidence_score_member = confidence_score(model, X_member, y_member)
      confidence_score_non_member = confidence_score(model, X_non_member, y_non_member)


  return negative_loss_member, negative_loss_non_member, confidence_score_member, confidence_score_non_member
