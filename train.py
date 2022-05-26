import tensorflow as tf
import callbacks
from config import config
import os
from tensorflow.keras.metrics import CategoricalAccuracy as tf_Accuracy

config_train_params = config['train_params']

#empty, only function names init and train

class Train:
    def __init__(self, model_object, train_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model_object

        models_dir = './models'
        self.model_name = config['model_name']
        self.checkpoint_dir_best = os.path.join(models_dir, self.model_name, "checkpoints_best")
        if not os.path.exists(self.checkpoint_dir_best):
            os.makedirs(self.checkpoint_dir_best)

        self.checkpoint_dir_latest = os.path.join(models_dir, self.model_name, "checkpoints_latest")
        if not os.path.exists(self.checkpoint_dir_latest):
            os.makedirs(self.checkpoint_dir_latest)

        self.summary_dir = os.path.join(models_dir, self.model_name, "summaries")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
            os.makedirs(os.path.join(self.summary_dir, 'train'))
            os.makedirs(os.path.join(self.summary_dir, 'dev'))

    def train(self):  # dev_dataset
        # making checkpoint and summary directories with their subdirs
        print('compiling the model')
        self.model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=[tf_Accuracy()], run_eagerly = False)

        # restore checkpoint from the latest
        ckpt_to_restore = tf.train.latest_checkpoint(self.checkpoint_dir_latest)
        if ckpt_to_restore:
            tf.train.Checkpoint(self.model).restore(ckpt_to_restore)

        best_checkpoint_callback = callbacks.BestWeightsSaver(checkpoint_dir=self.checkpoint_dir_best,
                                                              batch=self.model.optimizer.iterations.numpy())
        tboard = callbacks.LogMetricsCallback(self.dev_dataset, summary_dir=self.summary_dir,
                                              batch=self.model.optimizer.iterations.numpy())
        latest_checkpoint_callback = callbacks.LatestWeightsSaver(checkpoint_dir=self.checkpoint_dir_latest,
                                                                  batch=self.model.optimizer.iterations.numpy())

        try:
            print('fit started')
            self.model.fit(self.train_dataset,
                           epochs=config_train_params['epochs'],
                           steps_per_epoch=config_train_params['steps_per_epoch'],
                           callbacks=[tboard,latest_checkpoint_callback,best_checkpoint_callback],
                           verbose = 1)
        except KeyboardInterrupt:
            name = 'weights_%d' % self.model.optimizer.iterations.numpy()
            path_to_save = os.path.join(self.checkpoint_dir_latest, name)
            tf.train.Checkpoint(self.model).save(path_to_save)
