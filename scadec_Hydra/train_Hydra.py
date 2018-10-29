# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Modified on Feb, 2018 based on the work of jakeret

author: yusun
'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
from scadec_Hydra import util

from art import text2art, randart

SHORT_INFO = False
#from IPython.core.debugger import Tracer
#import ipdb
Y_RAND = False
SAVE_MODE = 'Xiaojian'# could be 'Xiaojian', 'Xing' or 'Original'
SAVE_TRAIN_PRED = True


class Trainer_bn(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    the phase of the unet are True by default

    """
    
    def __init__(self, net, batch_size=1, optimizer="adam", opt_kwargs={}, verbose = False):
        self.net = net
        self.batch_size = batch_size
        self.optimizer_type = optimizer # should be str
        self.opt_kwargs = opt_kwargs  
        self.verbose = verbose

    def _get_optimizer(self, training_iters, global_step):
        self.ifGAN = False
        if self.optimizer_type == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                                global_step=global_step)
            return optimizer
        
        elif self.optimizer_type == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                                global_step=global_step)
            return optimizer

        elif self.optimizer_type == "adam_clip":
            # Gradient clipping needs to happen after computing the gradients, but before applying them to update the model's parameters. In your example, both of those things are handled by the AdamOptimizer.minimize() method.
            # In order to clip your gradients you'll need to explicitly compute, clip, and apply them as described in this section in TensorFlow's API documentation. Specifically you'll need to substitute the call to the minimize() method with something like the following:
            # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow

            # "None values not supported.":
            # https://stackoverflow.com/questions/39295136/gradient-clipping-appears-to-choke-on-none
            # https://github.com/jazzsaxmafia/Inpainting/issues/6
            def ClipIfNotNone(grad):
                # If grad is None, don't clip
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -1, 1)

            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer_unclipped = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                                   **self.opt_kwargs)
                gvs = optimizer_unclipped.compute_gradients(self.net.loss)
                capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
                optimizer = optimizer_unclipped.apply_gradients(capped_gvs, global_step=global_step)
            
            return optimizer
        
        elif self.optimizer_type == "adam_patchPerceptual":
            self.ifGAN = True
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                                global_step=global_step)
                d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                                   **self.opt_kwargs).minimize(self.net.disc_loss,
                                                                                global_step=global_step)
            return g_optimizer, d_optimizer
        
        else:
            raise ValueError('Unknown optimizer: %s' % self.optimizer_type)
        
        
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)
        logging.getLogger().setLevel(logging.INFO)

        # get optimizer
        print("optimizer: %s" % self.optimizer_type)
        if self.optimizer_type != "adam_patchPerceptual":
            self.g_optimizer = self._get_optimizer(training_iters, global_step)
            # self.d_optimizer = tf.constant(tf.float32, 0)
        else:
            self.g_optimizer, self.d_optimizer = self._get_optimizer(training_iters, global_step)
        init = tf.global_variables_initializer()

        # get validation_path
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    def train(self, data_provider, output_path, valid_provider, valid_size, training_iters=100, epochs=1000, dropout=0.75, display_step=1, save_epoch=50, restore=False, write_graph=False, prediction_path='validation'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param valid_provider: data provider for the validation dataset
        :param valid_size: batch size for validation provider
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        
        # initialize the training process.
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        self.total_epochs = epochs

        # create output path
        directory = os.path.join(output_path, "final/")
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_path = os.path.join(directory, "model.cpkt")
        if epochs == 0:
            return save_path

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            # select validation dataset
            valid_x, valid_y, vbatch_cls = valid_provider(valid_size, fix=True)
            if SAVE_MODE == 'Original':
                util.save_mat(valid_y, "%s/%s.mat"%(self.prediction_path, 'origin_y'))
                util.save_mat(valid_x, "%s/%s.mat"%(self.prediction_path, 'origin_x'))
            # Xiaojian's code
            elif SAVE_MODE == 'Xiaojian':
                imgx = util.concat_n_images(valid_x)
                imgy = util.concat_n_images(valid_y)

                batch_cls_str = ['cls: '+ str(clas) for clas in list(vbatch_cls.argmax(1))]
                batch_outputs_psnr_str = ['PSNR: '+str(psnr) for psnr in list(util.computePSNRs(imgy, imgx))]
                batch_outputs_str = list(zip(batch_outputs_psnr_str, batch_cls_str))
                batch_targets_str = list(zip(['PSNR: Inf']*len(batch_cls_str),batch_cls_str))
                imgx = util.noteTexts2Imgs(imgx, batch_outputs_str)
                imgy = util.noteTexts2Imgs(imgy, batch_targets_str)

                util.save_img(imgx, "%s/%s_img.png"%(self.prediction_path, 'trainOb'))
                util.save_img(imgy, "%s/%s_img.png"%(self.prediction_path, 'trainGt'))

            for epoch in range(epochs):
                total_loss = 0
                # batch_x, batch_y = data_provider(self.batch_size)
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    # print('data_provider.onehot_cls')
                    # print(data_provider.onehot_cls)
                    batch_x, batch_y, batch_cls = data_provider(self.batch_size)
                    util.verbose_print('batch_cls in train', self.verbose)
                    util.verbose_print(batch_cls, self.verbose)

                    # print('Shape of batch_cls!')
                    # print(np.shape(batch_cls))
                    # Run optimization op (backprop)
                    if self.ifGAN:
                        _, _, loss_dict, lr, avg_psnr, train_output = sess.run([self.d_optimizer,
                                                            self.g_optimizer,
                                                            self.net.loss_dict,
                                                            self.learning_rate_node, 
                                                            self.net.avg_psnr,
                                                            self.net.recons], 
                                                            feed_dict={self.net.x: batch_x,
                                                                        self.net.y: batch_y,
                                                                        self.net.batch_cls: batch_cls,
                                                                        self.net.current_epoch: epoch,
                                                                        self.net.total_epochs: epochs,
                                                                        self.net.keep_prob: dropout,
                                                                        self.net.phase: True})
                    else:
                        _, loss_dict, lr, avg_psnr, train_output = sess.run([self.g_optimizer,
                                                            self.net.loss_dict,
                                                            self.learning_rate_node, 
                                                            self.net.avg_psnr,
                                                            self.net.recons], 
                                                            feed_dict={self.net.x: batch_x,
                                                                        self.net.y: batch_y,
                                                                        self.net.batch_cls: batch_cls,
                                                                        self.net.current_epoch: epoch,
                                                                        self.net.total_epochs: epochs,
                                                                        self.net.keep_prob: dropout,
                                                                        self.net.phase: True})
                    loss = loss_dict['total_loss']                    
                    
                    if step % display_step == 0:
                        # Changed here - Xing
                        # logging.info("Iter {:}".format(step))
                        logging.info("Iter {:} (before training on the batch)\tMinibatch MSE= {:.4f},\tMinibatch Avg PSNR= {:.4f}".format(step, loss, avg_psnr))
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, batch_y, batch_cls, epoch)
                        
                    total_loss += loss

                
                    for loss_name, loss_value in loss_dict.items():
                        self.record_summary(summary_writer, 'training_'+loss_name, loss_value, step)                
                    
                    self.record_summary(summary_writer, 'training_avg_psnr', avg_psnr, step)

                # output statistics for epoch
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.output_valstats(sess, summary_writer, step, valid_x, valid_y, vbatch_cls, "epoch_%s_valid"%epoch, epoch, store_img=True)
                # Xing
                if SAVE_TRAIN_PRED:
                    if SAVE_MODE == 'Original':
                        util.save_img(train_output[0,...], "%s/%s_img.tif"%(self.prediction_path, "epoch_%s_train"%epoch))
                    elif SAVE_MODE == 'Xiaojian':
                        # Xiaojian's code
                        self.output_train_batch_stats(sess, epoch, batch_x, batch_y, batch_cls, epoch)

                if epoch % save_epoch == 0:
                    directory = os.path.join(output_path, "{}_cpkt/".format(step))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    #path = os.path.join(directory, "model.cpkt".format(step))      
                    path = os.path.join(directory, "model.cpkt")
                    self.net.save(sess, path)

                save_path = self.net.save(sess, save_path)

            logging.info("Optimization Finished!")
            
            return save_path
    
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        if SHORT_INFO:
            logging.info("Epoch {:}".format(epoch))
        else:
            logging.info("Epoch {:}, Average MSE: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
        
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_cls, current_epoch):
        # Calculate batch loss and accuracy
        loss, predictions, avg_psnr = sess.run([self.net.loss,  
                                                self.net.recons,
                                                self.net.avg_psnr], 
                                                feed_dict={self.net.x: batch_x,
                                                            self.net.y: batch_y,
                                                            self.net.batch_cls: batch_cls,
                                                            self.net.current_epoch: current_epoch,
                                                            self.net.total_epochs: self.total_epochs,
                                                            self.net.keep_prob: 1.,
                                                            self.net.phase: False})

        self.record_summary(summary_writer, 'minibatch_loss', loss, step)
        self.record_summary(summary_writer, 'minibatch_avg_psnr', avg_psnr, step)

        # Xing
        if SHORT_INFO:
            logging.info("Iter {:}".format(step))
        else:
            logging.info("Iter {:} (After training on the batch)\tMinibatch MSE= {:.4f},\tMinibatch Avg PSNR= {:.4f}".format(step,loss,avg_psnr))

    def output_train_batch_stats(self, sess, epoch, batch_x, batch_y, batch_cls, current_epoch):
        # Xing
        # Calculate batch loss and accuracy
        loss, predictions, avg_psnr = sess.run([self.net.loss,  
                                                self.net.recons,
                                                self.net.avg_psnr], 
                                                feed_dict={self.net.x: batch_x,
                                                            self.net.y: batch_y,
                                                            self.net.batch_cls: batch_cls,
                                                            self.net.current_epoch: current_epoch,
                                                            self.net.total_epochs: self.total_epochs,
                                                            self.net.keep_prob: 1.,
                                                            self.net.phase: False})        
        train_inputs = util.concat_n_images(batch_x)
        train_outputs = util.concat_n_images(predictions)
        train_targets = util.concat_n_images(batch_y)
        
        batch_cls_str = ['cls: '+ str(clas) for clas in list(batch_cls.argmax(1))]
        batch_inputs_psnr_str = ['PSNR: '+str(psnr)  for psnr in list(util.computePSNRs(train_targets, train_inputs))]
        batch_outputs_psnr_str = ['PSNR: '+str(psnr) for psnr in list(util.computePSNRs(train_targets, train_outputs))]
        batch_inputs_str = list(zip(batch_inputs_psnr_str, batch_cls_str))
        batch_outputs_str = list(zip(batch_outputs_psnr_str, batch_cls_str))
        batch_targets_str = list(zip(['PSNR: Inf']*len(batch_cls_str),batch_cls_str))

        # img = util.noteTexts2Imgs(img, x_cls_str)
        train_inputs = util.noteTexts2Imgs(train_inputs, batch_inputs_str)
        train_outputs = util.noteTexts2Imgs(train_outputs, batch_outputs_str)
        train_targets = util.noteTexts2Imgs(train_targets, batch_targets_str)
        
        util.save_img(train_inputs, "%s/%s_img.png"%(self.prediction_path, "epoch_%s_train_inputs"%epoch))
        util.save_img(train_outputs, "%s/%s_img.png"%(self.prediction_path, "epoch_%s_train_outputs"%epoch))
        util.save_img(train_targets, "%s/%s_img.png"%(self.prediction_path, "epoch_%s_train_targets"%epoch))

    def output_valstats(self, sess, summary_writer, step, batch_x, batch_y, batch_cls, name, current_epoch, store_img=True):
        
        prediction, loss_dict, avg_psnr = sess.run([self.net.recons,
                                                self.net.valid_loss_dict,
                                                self.net.valid_avg_psnr], 
                                                feed_dict={self.net.x: batch_x, 
                                                            self.net.y: batch_y,
                                                            self.net.batch_cls: batch_cls,
                                                            self.net.current_epoch: current_epoch,
                                                            self.net.total_epochs: self.total_epochs,
                                                            self.net.keep_prob: 1.,
                                                            self.net.phase: False})
        loss = loss_dict['total_loss']
        for loss_name, loss_value in loss_dict.items():
            self.record_summary(summary_writer, 'valid_'+loss_name, loss_value, step)             
        
        self.record_summary(summary_writer, 'valid_avg_psnr', avg_psnr, step)
            
        # Xing
        if SHORT_INFO:
            logging.info("Iter {:}".format(step))
        else:
            # logging.info("Validation Statistics, validation loss= {:.4f}, Avg PSNR= {:.4f}".format(loss, avg_psnr))
            logging.info("Validation Statistics, validation loss= {:.4f}".format(loss))
            logging.info('\n'+text2art("Avg   PSNR:   {:.4f}".format(avg_psnr)))

        util.save_mat(prediction, "%s/%s.mat"%(self.prediction_path, name))

        if store_img:
            if SAVE_MODE == 'Original':
                util.save_img(prediction[0,...], "%s/%s_img.tif"%(self.prediction_path, name))
            elif SAVE_MODE == 'Xiaojian':
                # Xiaojian's code
                # img = util.concat_n_images(prediction)
                valid_outputs = util.concat_n_images(prediction)
                valid_targets = util.concat_n_images(batch_y)

                batch_cls_str = ['cls: '+ str(clas) for clas in list(batch_cls.argmax(1))]
                batch_outputs_psnr_str = ['PSNR: '+str(psnr) for psnr in list(util.computePSNRs(valid_targets, valid_outputs))]
                batch_outputs_str = list(zip(batch_outputs_psnr_str, batch_cls_str))

                valid_outputs = util.noteTexts2Imgs(valid_outputs, batch_outputs_str)

                # util.save_img(img, "%s/%s_img.tif"%(self.prediction_path, name))
                util.save_img(valid_outputs, "%s/%s_img.png"%(self.prediction_path, name))
            
            

    def record_summary(self, writer, name, value, step):
        summary=tf.Summary()
        #Tracer()
        # ipdb.set_trace()
        # Xing
        if SHORT_INFO:
            summary.value.add(tag=name, simple_value = np.mean(value))
        else:
            summary.value.add(tag=name, simple_value = value)
        writer.add_summary(summary, step)
        writer.flush()

