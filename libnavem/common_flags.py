from absl import flags
from absl import app

FLAGS = flags.FLAGS

#news FLAGS
flags.DEFINE_string("dataset", "dataset","Folder containing images and labels dataset experiments")
flags.DEFINE_string("exp_name", "exp_001","Folder containing models and results of experiments")


# Input
flags.DEFINE_integer('img_width', 224, 'Target Image Width')#320
flags.DEFINE_integer('img_height', 224, 'Target Image Height')#240

flags.DEFINE_integer('crop_img_width', 224, 'Cropped image widht')
flags.DEFINE_integer('crop_img_height', 224, 'Cropped image height')

flags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either rgb or grayscale')

# Training
flags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
flags.DEFINE_integer('epochs', 500, 'Number of epochs for training')
flags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
flags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')

# Files
flags.DEFINE_string('experiment_rootdir', "./model", 'Folder '
                     ' containing all the logs, model weights and results')
flags.DEFINE_string('train_dir', "./datasets/training", 'Folder containing'
                     ' training experiments')
flags.DEFINE_string('val_dir', "./datasets/validation", 'Folder containing'
                     ' validation experiments')
flags.DEFINE_string('test_dir', "./datasets/testing", 'Folder containing'
                     ' testing experiments')

# Model
flags.DEFINE_bool('restore_model', False, 'Whether to restore a trained model for training')
flags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) filename of model weights')
flags.DEFINE_string('json_model_fname', "model_struct.json", 'Model struct json serialization, filename')
