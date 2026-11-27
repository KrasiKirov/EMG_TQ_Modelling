"""Keras training of the windowed LSTM (ML.models.lstm) on plain MSE, with the schedule in ML.config.TRAINING."""

import numpy as np

from ML.config import CLIPNORM, MIN_LR, TRAINING


def fit_predict(data, units, layers, dropout, seed=0, lr=None, batch=None, save_weights=None, verbose=0):
    """Train on data['X_train'] / ['X_val'] and predict X_test; returns (y_pred_test (Nm), info).

    units, layers, dropout  the LSTM (ML.models.lstm.build_lstm)
    lr, batch               replace the schedule's learning rate and batch size
    save_weights            path for the best-epoch weights
    Windows are fed in their stored (time) order. Also returns the validation predictions (Nm, best-epoch weights)
    as info['val_pred'] and their MAE.
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Nadam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from ML.models.lstm import build_lstm

    tf.keras.utils.set_random_seed(seed)
    sch = TRAINING
    y_tr, y_va = data['y_train'].astype(np.float64), data['y_val'].astype(np.float64)

    model = build_lstm(data['X_train'].shape[1], data['X_train'].shape[2], units, layers, dropout)
    model.compile(optimizer=Nadam(learning_rate=lr or sch['lr'], clipnorm=CLIPNORM), loss='mse')
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=sch['lr_factor'], patience=sch['lr_patience'],
                             min_lr=MIN_LR, verbose=0),
           EarlyStopping(monitor='val_loss', patience=sch['early'], restore_best_weights=True, verbose=0)]
    # Keras 3 starts both from no best loss, so a NaN in epoch 1 would become the best and never be beaten;
    # starting from inf makes a NaN epoch count as no improvement (same as before when every loss is finite).
    for cb in cbs:
        cb.best = np.inf
    val = (data['X_val'], y_va.astype(np.float32))
    hist = model.fit(data['X_train'], y_tr.astype(np.float32),
                     validation_data=val, epochs=sch['epochs'], batch_size=batch or sch['batch'],
                     callbacks=cbs, shuffle=False, verbose=verbose)
    val_loss = hist.history['val_loss']
    if not np.isfinite(val_loss).any():
        raise FloatingPointError(f'validation loss is not finite in any of the {len(val_loss)} epochs')
    best = cbs[1].best_epoch                 # the epoch whose weights were restored; NaN epochs never count
    if save_weights:
        model.save_weights(save_weights)
    pred = model.predict(data['X_test'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    pv = model.predict(data['X_val'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    info = {'epochs': len(hist.history['loss']), 'best_epoch': best + 1,
            'best_val_loss': float(val_loss[best]),
            'n_params': int(model.count_params()),
            'lr': lr or sch['lr'],
            'updates': int(model.optimizer.iterations.numpy()),
            'clipnorm': CLIPNORM, 'optimizer_clipnorm': model.optimizer.clipnorm, 'max_epochs': sch['epochs'],
            'batch': batch or sch['batch'], 'shuffle': False, 'val_pred': pv, 'val_mae': float(np.mean(np.abs(pv - y_va)))}
    return pred, info
