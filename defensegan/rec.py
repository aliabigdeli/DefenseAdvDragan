# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""The main class for training GANs."""

import os
import argparse
import sys

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tflib

from models.gan import MnistDefenseGAN, FmnistDefenseDefenseGAN, \
    CelebADefenseGAN
from utils.config import load_config
from PIL import Image

def float2int(I):
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    return I8

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


def main(cfg, *args):
    FLAGS = tf.app.flags.FLAGS
    ds_gan = {
        'mnist': MnistDefenseGAN, 'f-mnist': FmnistDefenseDefenseGAN,
        'celeba': CelebADefenseGAN,
    }
    GAN = ds_gan[FLAGS.dataset_name]

    gan = GAN(cfg=cfg, test_mode=not FLAGS.is_train)

    gan.load_generator(ckpt_path=FLAGS.init_path)
    gan.sess.run(gan.global_step.initializer)
    batch_size = 1
    z_init_val = tf.constant(np.random.randn(batch_size * gan.rec_rr, gan.latent_dim).astype(np.float32))
    images = plt.imread('img/cwadv0.png')
    rec = gan.reconstruct(tf.convert_to_tensor(images.reshape((1, 28, 28, 1)), np.float32), batch_size=1, back_prop=True, reconstructor_id=0, z_init_val=z_init_val)
    print("########### rec type: {}".format(type(rec)))
    print("########### rec shape: {}".format(rec.get_shape()))

    with tf.Session() as sess:
      init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
      sess.run(init)
      init.run()
      recnd = rec.eval()
    print("########### recnd : {}".format(type(recnd)))
    print("########### recnd shape : {}".format(recnd.shape))
    # print("########### recnd : {}".format(recnd))
    
    # writer = tf.write_file('rec_img.png', rec)
    # sess.run(writer)

    imgs = Image.fromarray(float2int(recnd.reshape(28, 28)))
    imgs.save("my_rec2.png")

    # iteration = 1
    # tflib.save_images.save_images(
    #         recnd.reshape(28, 28),
    #         os.path.join('/content/drive/My Drive/Colab Notebooks/defensegan',
    #                      'rec_samples_{}.png'.format(iteration))
    #     )


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python train.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_boolean("is_train", False,
                         "True for training, False for testing. [False]")
    flags.DEFINE_boolean("save_recs", False,
                         "True for saving reconstructions. [False]")
    flags.DEFINE_boolean("debug", False,
                         "True for debug. [False]")
    flags.DEFINE_boolean("test_generator", False,
                         "True for generator samples. [False]")
    flags.DEFINE_boolean("test_decoder", False,
                         "True for decoder samples. [False]")
    flags.DEFINE_boolean("test_again", False,
                         "True for not using cache. [False]")
    flags.DEFINE_boolean("test_batch", False,
                         "True for visualizing the batches and labels. [False]")
    flags.DEFINE_boolean("save_ds", False,
                         "True for saving the dataset in a pickle file. ["
                         "False]")
    flags.DEFINE_boolean("tensorboard_log", True, "True for saving "
                                                  "tensorboard logs. [True]")
    flags.DEFINE_boolean("train_encoder", False,
                         "Add an encoder to a pretrained model. ["
                         "False]")
    flags.DEFINE_boolean("init_with_enc", False,
                         "Initializes the z with an encoder, must run "
                         "--train_encoder first. [False]")
    flags.DEFINE_integer("max_num", -1,
                         "True for saving the dataset in a pickle file ["
                         "False]")
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
