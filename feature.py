from preprocessing import Preprocessing
import tensorflow as tf
from config import config

config_feature = config['feature']
#init is given and function names
class FeatureWithMappings:
    def __init__(self):
        self.sample_rate = config['sample_rate']
        self.output_feature = config_feature['feature']
        # fft params
        window_size_ms = config_feature['window_size_ms']
        self.frame_length = int(self.sample_rate * window_size_ms)
        frame_step = config_feature['window_stride']
        self.frame_step = int(self.sample_rate * frame_step)
        assert (self.frame_step == self.sample_rate * frame_step), \
            'frame step,  must be integer '
        self.fft_length = config_feature['fft_length']
        # mfcc params
        self.lower_edge_hertz = config_feature['mfcc_lower_edge_hertz']
        self.upper_edge_hertz = config_feature['mfcc_upper_edge_hertz']
        self.num_mel_bins = config_feature['mfcc_num_mel_bins']
        # tf.debugging.set_log_device_placement(True)
        # with tf.device("/GPU:0"):
        self.linear_to_mel_weight_matrix = self.get_linear_to_mel_weight_matrix()

    def create_features(self, preprocessing):
       train_dataset, val_dataset, test_dataset = preprocessing.create_iterators()
       #do mapping
       return train_dataset, val_dataset, test_dataset
         
    def get_linear_to_mel_weight_matrix(self):
        num_spectrogram_bins = self.fft_length // 2 + 1
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, num_spectrogram_bins,
                                                                            self.sample_rate, self.lower_edge_hertz,
                                                                            self.upper_edge_hertz)
        return linear_to_mel_weight_matrix

    def get_stft(self, audio):
        stft = tf.signal.stft(audio, self.frame_length, self.frame_step, fft_length=512,  # self.frame_length,
                              window_fn=tf.signal.hamming_window)
        return stft

    def stft_to_log_mel_spectrogram(self, stft):
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(
            spectrogram.shape[:-1].concatenate(self.linear_to_mel_weight_matrix.shape[-1:]))  # ?
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram

    def map_input_to_mfcc(self, audio, label):  # (audio,label)
        stft = self.get_stft(audio)
        log_mel_spectrogram = self.stft_to_log_mel_spectrogram(stft)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        return [mfccs, label]



