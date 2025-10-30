import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import os
import random
from attack import empirical_auc_sklearn, empirical_advantage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import StratifiedKFold




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
    model.add(layers.InputLayer(shape=(input_dim,)))

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



def train_model(model, X_train, y_train, X_val, y_val,batch_size=128,epochs=30, use_early_stopping=True):

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

    y_probs = model.predict(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2,label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()




def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)       
    random.seed(seed)                               
    np.random.seed(seed)                             
    tf.random.set_seed(seed)        
    


def attack_evaluation_set(train_df, val_df, test_df, seed=42):
  #500 from train
  #500 from test and val combined
  val_test_df = pd.concat([val_df, test_df])

  member_df = train_df.sample(500, random_state = seed)
  non_member_df = val_test_df.sample(500, random_state = seed)

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





def get_input_gradients(model, x):

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        y_pred = model(x_tensor)
    
    grads = tape.gradient(y_pred, x_tensor)
    return grads



def grad_norm(grads):

  grads_np = grads.numpy() if isinstance(grads, tf.Tensor) else grads
  grad_norms = np.linalg.norm(grads_np, ord=2, axis=1)

  return grad_norms
                     

def seed_loop(target_df, target_col, drop_columns, K, early_stopping=True):

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
      
      
      train_model(model, X_train, y_train, X_val, y_val, use_early_stopping=early_stopping)
      evaluate_model(model, X_test, y_test, task_type='classification')
      classification_metrics(model, X_test, y_test)


      negative_loss_member = negative_loss(model, X_member, y_member)
      negative_loss_non_member = negative_loss(model, X_non_member, y_non_member)

      confidence_score_member = confidence_score(model, X_member, y_member)
      confidence_score_non_member = confidence_score(model, X_non_member, y_non_member)

      grads_member = get_input_gradients(model, X_member)
      grads_non_member = get_input_gradients(model, X_non_member)
      grads_non_member_norm = grad_norm(grads_non_member)
      grads_member_norm = grad_norm(grads_member)


  return negative_loss_member, negative_loss_non_member, confidence_score_member, confidence_score_non_member, grads_member_norm, grads_non_member_norm










def seed_loop_for_bootstrap2(target_df, target_col, drop_columns, K, early_stopping=True):

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


      model = build_mlp_model(input_dim=X_train.shape[1],
                              task_type='classification',
                              output_dim=K,
                              hidden_dim=256,
                              dropout_rate=0.3)
      
      
      model = compile_model(model, task_type='classification')     
      
      
      train_model(model, X_train, y_train, X_val, y_val, use_early_stopping=early_stopping)
      evaluate_model(model, X_test, y_test, task_type='classification')
      classification_metrics(model, X_test, y_test)

      
      confidence_auc = []
      grad_auc = []
      loss_auc = []
      for i in range(50):
      
        member_df, non_member_df = attack_evaluation_set(train_df, val_df, test_df, seed = i)
        X_member, y_member = X_y_split(member_df, drop_columns, target_col)
        X_non_member, y_non_member = X_y_split(non_member_df, drop_columns, target_col)


        negative_loss_member = negative_loss(model, X_member, y_member)
        negative_loss_non_member = negative_loss(model, X_non_member, y_non_member)

        confidence_score_member = confidence_score(model, X_member, y_member)
        confidence_score_non_member = confidence_score(model, X_non_member, y_non_member)

        grads_member = get_input_gradients(model, X_member)
        grads_non_member = get_input_gradients(model, X_non_member)
        grads_non_member_norm = grad_norm(grads_non_member)
        grads_member_norm = grad_norm(grads_member)
        

        confidence_auc.append(empirical_auc_sklearn(confidence_score_member, confidence_score_non_member))
        grad_auc.append(empirical_auc_sklearn(grads_member_norm, grads_non_member_norm))
        loss_auc.append(empirical_auc_sklearn(negative_loss_member, negative_loss_non_member))

      confidence_lower, confidence_upper = np.percentile(confidence_auc, [2.5, 97.5])
      grad_lower, grad_upper = np.percentile(grad_auc, [2.5, 97.5])
      loss_lower, loss_upper = np.percentile(loss_auc, [2.5, 97.5])


  return confidence_lower, confidence_upper, grad_lower, grad_upper, loss_lower, loss_upper






def dataset_creation(i, target_df, target_col, drop_columns):
  seed = i

  train_df, val_df = train_test_split(target_df, 0.6, SEED = seed)
  val_df, test_df = train_test_split(val_df, 0.5, SEED = seed)

  

  X_train, y_train = X_y_split(train_df, drop_columns, target_col)
  X_val, y_val = X_y_split(val_df, drop_columns, target_col)
  X_test, y_test = X_y_split(test_df, drop_columns, target_col)

  member_df, non_member_df = attack_evaluation_set(train_df, val_df, test_df)
  X_member, y_member = X_y_split(member_df, drop_columns, target_col)
  X_non_member, y_non_member = X_y_split(non_member_df, drop_columns, target_col)


  return X_train, y_train, X_val, y_val, X_test, y_test, X_member, y_member, X_non_member, y_non_member










def target_model_cv(df, target_col = 'disposition', drop_columns = ['disposition', 'subject_id', 'stay_id'], K = 2, early_stopping=True):

  
  X, y = X_y_split(df, drop_columns, target_col)
  
  n_splits = 5
  kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

  tprs = []      
  aucs = []      
  mean_fpr = np.linspace(0, 1, 100)

  plt.figure(figsize=(10, 8))

  for i, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
      
      model = build_mlp_model(input_dim=X_train.shape[1],
                              task_type='classification',
                              output_dim=K,
                              hidden_dim=256,
                              dropout_rate=0.3)

      model = compile_model(model, task_type='classification')  
      history = train_model(model, X_train, y_train, X_test, y_test, use_early_stopping=early_stopping)
      

      y_proba = model.predict(X_test)[:, 1]
      y_pred = (y_proba >= 0.5).astype(int)
      
      
      # ROC Curve and AUC
      fpr, tpr, _ = roc_curve(y_test, y_proba)
      roc_auc = auc(fpr, tpr)
      aucs.append(roc_auc)

      plt.plot(fpr, tpr, lw=2, label=f'Fold {i} (AUC = {roc_auc:.2f})')

      # Interpolation for plotting
      interp_tpr = np.interp(mean_fpr, fpr, tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)

      # Per-fold classification report
      print(classification_report(y_test, y_pred, digits=3))

  # Plot mean ROC
  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = np.mean(aucs)
  std_auc = np.std(aucs)
  plt.plot(mean_fpr, mean_tpr, label=f'Mean (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

  # Final plot settings
  plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
  plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
  plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
  plt.title('ROC Curve Comparison (5-Fold CV)', fontsize=16, fontweight='bold')
  plt.legend(loc='lower right', fontsize=14)
  plt.xticks(fontsize=14, fontweight='bold')
  plt.yticks(fontsize=14, fontweight='bold')
  plt.grid(True)
  plt.tight_layout()
  plt.show()
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

        
        

