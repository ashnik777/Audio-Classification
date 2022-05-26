config = {
    'data_dir': '../data/speech_commands',
    'model_name': '1',
    'sample_rate': 8000,
    'input_len': 16000,
    'NUM_PARALLEL_CALLS': 32,
    'preprocessing': {
        'noise_augmentation': True,
        'noise_mixing_probability': 0.8,
        'dbs': [0.0, 5.0, 10.0]
    },
    'feature': {
        'window_size_ms': 0.025,
        'window_stride': 0.01,
        'fft_length': 512,
        'mfcc_lower_edge_hertz': 0.0,
        'mfcc_upper_edge_hertz': 4000.0,  
        'mfcc_num_mel_bins': 64
    },
    'train_params': {
        'batch_size': 512,
        'epochs': 1000,
        'steps_per_epoch': None,
        'latest_checkpoint_step': 2,
        'validation_step': 2, #also summary step
        'max_checkpoints_to_keep': 5,
    },
    'matchbox': {
        'B': 3,
        'R': 1,
        'C': 64,
        'kernel_sizes': [13, 15, 17],
        'dropout': 0.2,
        'data_normalization': True,
        'val_accuracy_intial_max_value': 0.0002,
        'validation_accuracy_decay_rate': 0.98
    }

}
#config is fully given in stead of matchbox
# model: {
#     'dropout': 0.2,
#     'data_normalization': True,
#     'val_accuracy_intial_max_value': 0.0002,
#     'validation_accuracy_decay_rate': 0.98
# }