import os
from random import shuffle
import tensorflow as tf
import glob
from config import config

# all functions except init and create_iterators should be empty
class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_name = config['data_dir']
        self.input_len = config['input_len']
        #maybe add later noise augmentation
        
    def create_iterators(self):
        # get the filenames split into train test validation
        test_files = self.get_files_from_txt('testing_list.txt')
        val_files = self.get_files_from_txt('validation_list.txt')
        filenames = glob.glob(os.path.join(self.dir_name, '*/**.wav'), recursive=True)
        filenames = [filename for filename in filenames if 'background_noise' not in filename]
        train_files = list(set(filenames) - set(val_files) - set(test_files))
        shuffle(train_files)
        # get the commands and some prints
        self.commands = self.get_commands()
        self.num_classes = len(self.commands)
        print('len(train_data)', len(train_files))
        print('prelen(test_data)', len(test_files))
        print('len(val_data)', len(val_files))
        print('commands: ', self.commands)
        print('number of commands: ', len(self.commands))

        # make tf dataset object
        train_dataset = self.make_tf_dataset_from_list(train_files)
        val_dataset = self.make_tf_dataset_from_list(val_files, is_validation = True)
        test_dataset = self.make_tf_dataset_from_list(test_files)
        return train_dataset, val_dataset, test_dataset

    def get_files_from_txt(self, which_txt):
        assert which_txt == 'testing_list.txt' or which_txt == 'validation_list.txt', 'wrong argument'
        path = os.path.join(self.dir_name, which_txt)
        with open(path) as f:
            paths = f.readlines()
        paths = [os.path.join(self.dir_name, path[:len(path) - 1]) for path in paths]
        shuffle(paths)
        return paths

    def get_commands(self):
        dirs = glob.glob(os.path.join(self.dir_name, "*", ""))
        commands = [os.path.split(os.path.split(dir)[0])[1] for dir in dirs if 'background' not in dir]
        return commands

    @staticmethod
    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.commands)
        label = tf.one_hot(label_id, self.num_classes)
        return label

    def make_tf_dataset_from_list(self, filenames_list, is_validation = False):
        files = tf.data.Dataset.from_tensor_slices(filenames_list)
        dataset = files.map(self.get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.pad_map_func, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True) 
        if is_validation:
            dataset = dataset.repeat()
        dataset = dataset.batch(config['train_params']['batch_size']).prefetch(tf.data.AUTOTUNE)            
        return dataset

    def get_waveform_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label

    def pad_map_func(self, audio, label):
        return [self.add_paddings(audio), label]

    def add_paddings(self, wav):
        len_wav = len(wav)
        if len_wav < self.input_len:
            paddings = tf.zeros((self.input_len - len_wav))
            wav = tf.concat([wav, paddings], axis=0)
        return wav
