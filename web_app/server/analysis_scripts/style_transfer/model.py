from __future__ import division
import os
import time
from shutil import copyfile
from glob import glob
import tensorflow as tf
import numpy as np
import config
from collections import namedtuple
from module import *
from utils import *
from ops import *
from metrics import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size  # cropped size
        self.time_step = args.time_step
        self.pitch_range = args.pitch_range
        self.input_c_dim = args.input_nc  # number of input image channels
        self.output_c_dim = args.output_nc  # number of output image channels
        self.L1_lambda = args.L1_lambda
        self.gamma = args.gamma
        self.sigma_d = args.sigma_d
        
        self.model = args.model
        self.discriminator = discriminator
        self.generator = generator_resnet
        self.criterionGAN = mae_criterion

        if args.sourceStyle == 'baroque' and args.targetStyle == 'classical':
            self.load_dir = './checkpoint/baroque2classical'
            self.which_direction = 'AtoB'
        elif args.sourceStyle == 'classical' and args.targetStyle == 'baroque':
            self.load_dir = './checkpoint/baroque2classical'
            self.which_direction = 'BtoA'
        elif args.sourceStyle == 'baroque' and args.targetStyle == 'romantic':
            self.load_dir = './checkpoint/baroque2romantic'
            self.which_direction = 'AtoB'
        elif args.sourceStyle == 'romantic' and args.targetStyle == 'baroque':
            self.load_dir = './checkpoint/baroque2romantic'
            self.which_direction = 'BtoA'
        elif args.sourceStyle == 'baroque' and args.targetStyle == 'modern':
            self.load_dir = './checkpoint/baroque2modern'
            self.which_direction = 'AtoB'
        elif args.sourceStyle == 'modern' and args.targetStyle == 'baroque':
            self.load_dir = './checkpoint/baroque2modern'
            self.which_direction = 'BtoA'
        elif args.sourceStyle == 'classical' and args.targetStyle == 'romantic':
            self.load_dir = './checkpoint/classical2romantic'
            self.which_direction = 'AtoB'
        elif args.sourceStyle == 'romantic' and args.targetStyle == 'classical':
            self.load_dir = './checkpoint/classical2romantic'
            self.which_direction = 'BtoA'
        elif args.sourceStyle == 'classical' and args.targetStyle == 'modern':
            self.load_dir = './checkpoint/classical2modern'
            self.which_direction = 'AtoB'
        elif args.sourceStyle == 'modern' and args.targetStyle == 'classical':
            self.load_dir = './checkpoint/classical2modern'
            self.which_direction = 'BtoA'
        elif args.sourceStyle == 'romantic' and args.targetStyle == 'modern':
            self.load_dir = './checkpoint/romantic2modern'
            self.which_direction = 'AtoB'
        elif args.sourceStyle == 'modern' and args.targetStyle == 'romantic':
            self.load_dir = './checkpoint/romantic2modern'
            self.which_direction = 'BtoA'
        else:
            print('No such style transfer model')
            print('Please choose from baroque, classical, romantic, modern')
            return
        





        OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                        'image_size '
                                        'gf_dim '
                                        'df_dim '
                                        'output_c_dim '
                                        'is_training')
        self.options = OPTIONS._make((args.batch_size,
                                      args.fine_size,
                                      args.ngf,
                                      args.ndf,
                                      args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=30)
        # self.now_datetime = get_now_datetime()
        self.now_datetime = '2024-04-02'
        self.pool = ImagePool(args.max_size)

    def _build_model(self):

        # define some placeholders
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                     self.input_c_dim + self.output_c_dim], name='real_A_and_B')
        if self.model != 'base':
            self.real_mixed = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                          self.input_c_dim], name='real_A_and_B_mixed')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.gaussian_noise = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                          self.input_c_dim], name='gaussian_noise')
        # Generator: A - B - A
        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        # Generator: B - A - B
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        # to binary
        self.real_A_binary = to_binary(self.real_A, 0.5)
        self.real_B_binary = to_binary(self.real_B, 0.5)
        self.fake_A_binary = to_binary(self.fake_A, 0.5)
        self.fake_B_binary = to_binary(self.fake_B, 0.5)
        self.fake_A__binary = to_binary(self.fake_A_, 0.5)
        self.fake_B__binary = to_binary(self.fake_B_, 0.5)

        # Discriminator: Fake
        self.DB_fake = self.discriminator(self.fake_B + self.gaussian_noise, self.options,
                                          reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A + self.gaussian_noise, self.options,
                                          reuse=False, name="discriminatorA")
        # Discriminator: Real
        self.DA_real = self.discriminator(self.real_A + self.gaussian_noise, self.options, reuse=True,
                                          name="discriminatorA")
        self.DB_real = self.discriminator(self.real_B + self.gaussian_noise, self.options, reuse=True,
                                          name="discriminatorB")

        self.fake_A_sample = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                         self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                         self.input_c_dim], name='fake_B_sample')
        self.DA_fake_sample = self.discriminator(self.fake_A_sample + self.gaussian_noise,
                                                 self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample + self.gaussian_noise,
                                                 self.options, reuse=True, name="discriminatorB")
        
        # Generator loss
        self.cycle_loss = self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.cycle_loss
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.cycle_loss
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a - self.cycle_loss
        # Discriminator loss
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        # Define all summaries
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.cycle_loss_sum = tf.summary.scalar("cycle_loss", self.cycle_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum, self.cycle_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
                                        self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                                        self.d_loss_sum])

        # Test
        self.test_A = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range,
                                                  self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range,
                                                  self.output_c_dim], name='test_B')
        # A - B - A
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA_ = self.generator(self.testB, self.options, True, name='generatorB2A')
        # B - A - B
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")
        self.testB_ = self.generator(self.testA, self.options, True, name='generatorA2B')
        # to binary
        self.test_A_binary = to_binary(self.test_A, 0.5)
        self.test_B_binary = to_binary(self.test_B, 0.5)
        self.testA_binary = to_binary(self.testA, 0.5)
        self.testB_binary = to_binary(self.testB, 0.5)
        self.testA__binary = to_binary(self.testA_, 0.5)
        self.testB__binary = to_binary(self.testB_, 0.5)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]



    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        print('Checkpoint_dir: ', checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        song = np.load(args.input_path)
        print(song.shape)
        
        if self.load(self.load_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if self.which_direction == 'AtoB':
            out_var, in_var = (self.testB_binary, self.test_A)
        else:
            out_var, in_var = (self.testA_binary, self.test_B)

        transfer = self.sess.run(out_var, feed_dict={in_var: song * 1.})
        save_midis(transfer, args.output_path, 80)
        # np.save('./Dataset/testing_song/lamb_transfer.npy', transfer)