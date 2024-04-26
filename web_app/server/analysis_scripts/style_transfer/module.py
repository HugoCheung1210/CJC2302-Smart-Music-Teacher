from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # (32, 42, 64)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 4, name='d_h1_conv'), 'd_bn1'))
        # (16, 21, 256)
        h4 = conv2d(h1, 1, s=1, name='d_h3_pred')
        # (16, 21, 1)
        return h4

def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            # e.g, x is (# of images * 128 * 128 * 3)
            p = int((ks - 1) / 2)
            # For ks = 3, p = 1
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            # After first padding, (# of images * 130 * 130 * 3)
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            # After first conv2d, (# of images * 128 * 128 * 3)
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            # After second padding, (# of images * 130 * 130 * 3)
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            # After second conv2d, (# of images * 128 * 128 * 3)
            return relu(y + x)


        # Original image is (# of images * 256 * 256 * 3)
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # c0 is (# of images * 262 * 262 * 3)
        c1 = relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        # c1 is (# of images * 256 * 256 * 64)
        c2 = relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        # c2 is (# of images * 128 * 128 * 128)
        c3 = relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # c3 is (# of images * 64 * 64 * 256)

        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        # r1 is (# of images * 64 * 64 * 256)
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        # r2 is (# of images * 64 * 64 * 256)
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        # r3 is (# of images * 64 * 64 * 256)
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        # r4 is (# of images * 64 * 64 * 256)
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        # r5 is (# of images * 64 * 64 * 256)
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        # r6 is (# of images * 64 * 64 * 256)
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        # r7 is (# of images * 64 * 64 * 256)
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        # r8 is (# of images * 64 * 64 * 256)
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
        # r9 is (# of images * 64 * 64 * 256)
        r10 = residule_block(r9, options.gf_dim*4, name='g_r10')

        d1 = relu(instance_norm(deconv2d(r10, options.gf_dim*2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
        # d1 is (# of images * 128 * 128 * 128)
        d2 = relu(instance_norm(deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc'), 'g_d2_bn'))
        # d2 is (# of images * 256 * 256 * 64)
        d3 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # After padding, (# of images * 262 * 262 * 64)
        pred = tf.nn.sigmoid(conv2d(d3, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))
        # Output image is (# of images * 256 * 256 * 3)

        return pred


def PhraseGenerator(in_tensor, output_dim, reuse=False, name='generator'):

    with tf.variable_scope(name, reuse=reuse):
        h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
        h0 = relu(batch_norm(deconv2d_musegan(tensor_in=h0,
                                              out_shape=[2, 1],
                                              out_channels=1024,
                                              kernels=[2, 1],
                                              strides=[2, 1],
                                              name='h1'),
                             'h1_bn'))
        h1 = relu(batch_norm(deconv2d_musegan(tensor_in=h0,
                                              out_shape=[4, 1],
                                              out_channels=output_dim,
                                              kernels=[3, 1],
                                              strides=[1, 1],
                                              name='h2'),
                             'h2_bn'))
        h1 = tf.transpose(tf.squeeze(h1, axis=2), [0, 2, 1])

    return h1


def BarGenerator(in_tensor, output_dim, reuse=False, name='generator'):

    with tf.variable_scope(name, reuse=reuse):

        h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
        h0 = relu(batch_norm(deconv2d_musegan(h0, [1, 1], 1024, kernels=[1, 1], strides=[1, 1], name='h0'), 'h0_bn'))

        h1 = tf.reshape(h0, [-1, 2, 1, 512])
        h1 = relu(batch_norm(deconv2d_musegan(h1, [4, 1], 512, kernels=[2, 1], strides=[2, 1], name='h1'), 'h1_bn'))

        h2 = relu(batch_norm(deconv2d_musegan(h1, [8, 1], 256, kernels=[2, 1], strides=[2, 1], name='h2'), 'h2_bn'))

        h3 = relu(batch_norm(deconv2d_musegan(h2, [16, 1], 256, kernels=[2, 1], strides=[2, 1], name='h3'), 'h3_bn'))

        h4 = relu(batch_norm(deconv2d_musegan(h3, [32, 1], 128, kernels=[2, 1], strides=[2, 1], name='h4'), 'h4_bn'))

        h5 = relu(batch_norm(deconv2d_musegan(h4, [96, 1], 128, kernels=[3, 1], strides=[3, 1], name='h5'), 'h5_bn'))

        h6 = relu(batch_norm(deconv2d_musegan(h5, [96, 7], 64, kernels=[1, 7], strides=[1, 1], name='h6'), 'h6_bn'))

        h7 = deconv2d_musegan(h6, [96, 84], output_dim, kernels=[1, 12], strides=[1, 12], name='h7')

    return tf.nn.tanh(h7)


def BarDiscriminator(in_tensor, reuse=False, name='discriminator'):

    with tf.variable_scope(name, reuse=reuse):

        ## conv
        h0 = lrelu(conv2d_musegan(in_tensor, 128, kernels=[1, 12], strides=[1, 12], name='h0'))
        h1 = lrelu(conv2d_musegan(h0, 128, kernels=[1, 7], strides=[1, 7], name='h1'))
        h2 = lrelu(conv2d_musegan(h1, 128, kernels=[2, 1], strides=[2, 1], name='h2'))
        h3 = lrelu(conv2d_musegan(h2, 128, kernels=[2, 1], strides=[2, 1], name='h3'))
        h4 = lrelu(conv2d_musegan(h3, 256, kernels=[4, 1], strides=[2, 1], name='h4'))
        h5 = lrelu(conv2d_musegan(h4, 512, kernels=[3, 1], strides=[2, 1], name='h5'))

        ## linear
        h6 = tf.reshape(h5, [-1, np.product([s.value for s in h5.get_shape()[1:]])])
        h6 = lrelu(linear(h6, 1024, scope='h6'))
        h7 = linear(h6, 1, scope='h7')

    return h5, h7


def PhraseDiscriminator(in_tensor, reuse=False, name='discriminator'):

    with tf.variable_scope(name, reuse=reuse):

        ## conv
        h0 = lrelu(conv2d_musegan(tf.expand_dims(in_tensor, axis=2), 512, kernels=[2, 1], strides=[1, 1], name='h0'))
        h1 = lrelu(conv2d_musegan(h0, 128, kernels=[3, 1], strides=[3, 1], name='h1'))

        ## linear
        h2 = tf.reshape(h1, [-1, np.product([s.value for s in h1.get_shape()[1:]])])
        h2 = lrelu(linear(h2, 1024, scope='h2'))
        h3 = linear(h2, 1, scope='h3')

    return h3
