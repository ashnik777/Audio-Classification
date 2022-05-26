import tensorflow as tf
from config import config
config_feature = config['feature']

class DataNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(DataNormalization, self).__init__(name='data_normalization')  #
        self.n = tf.Variable(initial_value=1.0, trainable=False, name='batch_number')

    def build(self, input_shape):
        self.moving_myu = tf.Variable(initial_value=tf.zeros([input_shape[-1], ]), trainable=False,
                                      name='data_norm_moving_myu')
        self.moving_sigma = tf.Variable(initial_value=tf.ones([input_shape[-1], ]), trainable=False,
                                        name='data_norm_moving_sigma')

    def call(self, input_tensor, training):
        new_myu = tf.math.reduce_mean(input_tensor, axis=(0, 1))
        new_sigma = tf.math.reduce_mean(input_tensor ** 2, axis=(0, 1)) - new_myu ** 2
        if training:
            if self.n == 1.0:
                self.moving_myu.assign(new_myu)
                self.moving_sigma.assign(new_sigma)
            else:
                self.moving_myu.assign_add((new_myu - self.moving_myu) / self.n)
                self.moving_sigma.assign_add((new_sigma - self.moving_sigma) / self.n)

            x = (input_tensor - new_myu) / new_sigma
        else:  # test validation
            x = (input_tensor - self.moving_myu) / self.moving_sigma
        self.n.assign_add(1.0)
        return x

class FeatureLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureLayer, self).__init__(trainable=False)
        self.sample_rate = config['sample_rate']
        window_size_ms = config_feature['window_size_ms']
        self.frame_length = int(self.sample_rate * window_size_ms)
        frame_step = config_feature['window_stride']
        self.frame_step = int(self.sample_rate * frame_step)
        self.fft_length = config_feature['fft_length']
        self.lower_edge_hertz = config_feature['mfcc_lower_edge_hertz']
        self.upper_edge_hertz = config_feature['mfcc_upper_edge_hertz']
        self.num_mel_bins = config_feature['mfcc_num_mel_bins']
        self.linear_to_mel_weight_matrix = self.get_linear_to_mel_weight_matrix()

    def call(self, input_tensor):
        stft = tf.signal.stft(input_tensor, self.frame_length, self.frame_step, fft_length=self.fft_length,
                              window_fn=tf.signal.hamming_window)
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1, name='saten')
        mel_spectrogram.set_shape(
            spectrogram.shape[:-1].concatenate(self.linear_to_mel_weight_matrix.shape[-1:]))  # ?
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        return mfccs

    def get_linear_to_mel_weight_matrix(self):
        num_spectrogram_bins = self.fft_length // 2 + 1
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, num_spectrogram_bins,
                                                                            self.sample_rate, self.lower_edge_hertz,
                                                                            self.upper_edge_hertz)
        return linear_to_mel_weight_matrix
