import numpy as np

opts = {}

opts['batch_size'] = 16 
opts['gpu_ids'] = ["0"]
opts['log_dir'] = 'log_dir'
opts['n_summary'] = 40  # number of iterations after which to run the summary-op
opts['n_test'] = 2000
opts['n_checkpoint'] = 1000 # number of iteration after which to save the model
opts['image_size'] = 224
opts['channel_wise_fc'] = False
opts['preds_2d'] = False
opts['sk_3d'] = True
opts['z_embed'] = False
opts['n_joints'] = 17
opts['num_cam_angles'] = 3  # 3 angles
opts['num_cam_params'] = 3  # 0 2d translation, 3 translation
opts['translation_param'] = 0
opts['flip_prob'] = 0.4
opts['tilt_prob'] = 0.3
opts['tilt_limit'] = 10
opts['jitter_prob'] = 0.3
opts['procustes'] = True

