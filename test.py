import os
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy as tf_Accuracy
from config import config
 #empty
#only comments
class Test:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self, checkpoint_path_arg):
        model_name = config['model_name']
        models_dir = './models'
        if checkpoint_path_arg == 'latest' or 'best':
            dir_name = 'checkpoints_' + checkpoint_path_arg
            checkpoint_dir = os.path.join(models_dir, model_name, dir_name)
            assert os.path.exists(checkpoint_dir), 'checkpoint directory does not exist'
            checkpoint_path_arg = tf.train.latest_checkpoint(checkpoint_dir)

        tf.train.Checkpoint(self.model).restore(checkpoint_path_arg)

        self.model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=[tf_Accuracy()])
        loss, acc = self.model.evaluate(self.dataset)
        print(f'Loss: {loss}, Accuracy: {acc}')
        # 
        # accuracies = []
        # for batch in self.dataset:
        #     proba = self.model.predict(batch[0])
        #     a = proba.argmax(axis = 1)
        #     # print(proba.shape,a.shape,a)
        #     predictions = tf.math.argmax(batch[1],axis = 1)
        #     # print(predictions)
        #     f = a == predictions
        #     a = tf.reduce_sum(tf.cast(f, tf.float32))
        #     accuracies.append(a/512)
        # a = sum(accuracies)/166
       
        # print('accuracy', a)
