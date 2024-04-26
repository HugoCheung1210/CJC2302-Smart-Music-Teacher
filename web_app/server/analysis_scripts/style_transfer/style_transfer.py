import argparse
import os
import tensorflow as tf
from model import cyclegan
from style_classifier import Classifer
tf.set_random_seed(19)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--time_step', dest='time_step', type=int, default=64, help='time step of pianoroll')
parser.add_argument('--pitch_range', dest='pitch_range', type=int, default=84, help='pitch range of pianoroll')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--gamma', dest='gamma', type=float, default=1.0, help='weight of extra discriminators')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--sigma_d', dest='sigma_d', type=float, default=1.0, help='sigma of gaussian noise of discriminators')
parser.add_argument('--sourceStyle', dest='sourceStyle', help='Original style of music era')
parser.add_argument('--targetStyle', dest='targetStyle', help='Target transfer style of music era')
parser.add_argument('--input_path', dest='input_path', help='input of midi file')
parser.add_argument('--output_path', dest='output_path', help='output path of the generated midi file')


args = parser.parse_args()


def main(_):

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.visible_device_list = '0'
    if tf.test.is_gpu_available():
        print("GPU available.")
    else:
        print("No GPU available.")
    with tf.Session(config=tfconfig) as sess:

            print("Start generating")

            model = cyclegan(sess, args)
            model.test(args)

     


if __name__ == '__main__':
    tf.app.run()
