from tensorflow import keras
from config import config
import tensorflow as tf
import os
import glob
import re
config_train_params = config['train_params']

#only function names and comments

class LogMetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_data, summary_dir, batch):
        self.freq = config_train_params['validation_step']
        self.summary_dir = summary_dir
        self.batch = batch
        self.val_data_iter = iter(val_data)
        self.train_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))
        self.dev_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'dev'))
        self.alpha = config['matchbox']['validation_accuracy_decay_rate']

    def on_train_batch_end(self, batch, logs): 
        if self.batch % self.freq == 0:
            val_data = next(self.val_data_iter)
            val_loss, val_acc = self.model.evaluate(val_data[0], val_data[1], batch_size=config_train_params['batch_size'],
                                                     verbose=0)
            with self.train_writer.as_default():
                tf.summary.scalar('loss', data=logs['loss'], step=self.batch)
                tf.summary.scalar('categorical_accuracy:', data=logs['categorical_accuracy'], step=self.batch)
            with self.dev_writer.as_default():
                tf.summary.scalar('loss', data=val_loss, step=self.batch)
                tf.summary.scalar('categorical_accuracy:', data=val_acc, step=self.batch)
            
            # self.model.val_accuracy_moving_mean.assign(self.alpha * self.model.val_accuracy_moving_mean + (
            #         1 - self.alpha) * val_acc)
        self.batch += 1


class LatestWeightsSaver(keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, batch, max_to_keep=config_train_params['max_checkpoints_to_keep']):
        self.ckpt_step = config_train_params['latest_checkpoint_step']
        self.batch = batch
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep

    @staticmethod
    def get_numbers(string):
        regex = re.compile(r'weights_\d+')
        x = regex.findall(string)[0]
        regex1 = re.compile(r'\d+')
        x = regex1.findall(x)[0]
        return int(x)

    def get_largest_ckpt_number(self, list_ckpt_paths):
        numbers = list(map(self.get_numbers, list_ckpt_paths))
        # print('numbers', numbers)
        numbers.sort()
        largest = numbers[-1]
        # print('largest',largest)
        return largest

    def on_train_batch_end(self, batch, logs={}):
        if self.batch % self.ckpt_step == 0:
            name = 'weights_%d' % self.batch
            path = os.path.join(self.checkpoint_dir, name)
            tf.train.Checkpoint(self.model).save(path)
            list_ckpt_paths = glob.glob(os.path.join(self.checkpoint_dir, '*.index'))
            num_checkpoints = len(list_ckpt_paths)
            if num_checkpoints > self.max_to_keep:
                last_ckpt = self.get_largest_ckpt_number(list_ckpt_paths)
                to_delete_ckpt = last_ckpt - self.max_to_keep*self.ckpt_step
                name = 'weights_%d-1' % to_delete_ckpt
                paths_to_delete = glob.glob(os.path.join(self.checkpoint_dir, name + '.*'))
                for path in paths_to_delete:
                    os.remove(path)      
        self.batch += 1


class BestWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, batch):
        self.ckpt_step = config_train_params['validation_step']
        self.batch = batch
        self.checkpoint_dir = checkpoint_dir
        self.alpha = config['matchbox']['validation_accuracy_decay_rate']


    def on_train_batch_end(self, batch, logs={}):
        if self.batch % self.ckpt_step == 0:
            name = 'weights_%d' % self.batch
            path_to_save = os.path.join(self.checkpoint_dir, name)
            n = self.batch / self.ckpt_step + 1
            S_n_corrected = self.model.val_accuracy_moving_mean/(1-self.alpha**n)
            # tf.print('\n S_n_corrected',S_n_corrected,'self.model.val_accuracy_moving_mean_max',self.model.val_accuracy_moving_mean_max)
            if S_n_corrected > self.model.val_accuracy_moving_mean_max:
                tf.print('S_n_corrected',S_n_corrected)
                paths_to_delete = glob.glob(os.path.join(self.checkpoint_dir, '*.*'))
                for path_to_delete in paths_to_delete:
                    os.remove(path_to_delete)
                tf.train.Checkpoint(self.model).save(path_to_save)
                self.model.val_accuracy_moving_mean_max.assign(S_n_corrected)
        self.batch += 1

