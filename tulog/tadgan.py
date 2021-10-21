# -*- coding: utf-8 -*-

import logging
import math
import tempfile
from functools import partial

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.models import Model
from mlprimitives.adapters.keras import build_layer
from mlprimitives.utils import import_object
from scipy import stats

#from orion.primitives.timeseries_errors import reconstruction_errors
from timeseries_errors import reconstruction_errors

#from tensorflow import keras

LOGGER = logging.getLogger(__name__)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        """
        Args:
            inputs[0] x     original input
            inputs[1] x_    predicted input
        """
        alpha = K.random_uniform((bbb, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class TadGAN(object):
    """TadGAN model for time series reconstruction.

    Args:
        shape (tuple):
            Tuple denoting the shape of an input sample.
        encoder_input_shape (tuple):
            Shape of encoder input.
        generator_input_shape (tuple):
            Shape of generator input.
        critic_x_input_shape (tuple):
            Shape of critic_x input.
        critic_z_input_shape (tuple):
            Shape of critic_z input.
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        layers_critic_x (list):
            List containing layers of critic_x.
        layers_critic_z (list):
            List containing layers of critic_z.
        optimizer (str):
            String denoting the keras optimizer.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 2000.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        batch_size (int):
            Integer denoting the batch size. Default 64.
        iterations_critic (int):
            Optional. Integer denoting the number of critic training steps per one
            Generator/Encoder training step. Default 5.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """

    def __getstate__(self):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        modules = ['optimizer', 'critic_x_model', 'critic_z_model', 'encoder_generator_model']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = keras.models.load_model(fd.name)

        self.__dict__ = state

    def _build_model(self, hyperparameters, layers, input_shape):
        x = Input(shape=input_shape)
        model = keras.models.Sequential()

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        #print("hyperparameters: ", hyperparameters)
        #print(Model(x, model(x)).__dict__)
        #print()
        #print()
        return Model(x, model(x))

    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def __init__(self, shape, encoder_input_shape, generator_input_shape, critic_x_input_shape,
                 critic_z_input_shape, layers_encoder, layers_generator, layers_critic_x,
                 layers_critic_z, optimizer, learning_rate=0.0005, epochs=2000, latent_dim=20,
                 batch_size=64, iterations_critic=5, **hyperparameters):

        self.shape = shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        global bbb
        bbb = batch_size
        self.iterations_critic = iterations_critic
        self.epochs = epochs
        self.hyperparameters = hyperparameters

        self.encoder_input_shape = encoder_input_shape
        self.generator_input_shape = generator_input_shape
        self.critic_x_input_shape = critic_x_input_shape
        self.critic_z_input_shape = critic_z_input_shape

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        self.optimizer = import_object(optimizer)(learning_rate)
        
        self.total_cx_loss = []
        self.total_cz_loss = []
        self.total_g_loss = []

    def _build_tadgan(self, **kwargs):

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        self.encoder = self._build_model(hyperparameters, self.layers_encoder,
                                         self.encoder_input_shape)
        self.generator = self._build_model(hyperparameters, self.layers_generator,
                                           self.generator_input_shape)
        self.critic_x = self._build_model(hyperparameters, self.layers_critic_x,
                                          self.critic_x_input_shape)
        self.critic_z = self._build_model(hyperparameters, self.layers_critic_z,
                                          self.critic_z_input_shape)

        self.generator.trainable = False
        self.encoder.trainable = False

        z = Input(shape=(self.latent_dim, 1))
        x = Input(shape=self.shape)
        x_ = self.generator(z)
        z_ = self.encoder(x)
        fake_x = self.critic_x(x_)
        valid_x = self.critic_x(x)
        
        interpolated_x = RandomWeightedAverage()([x, x_])
        validity_interpolated_x = self.critic_x(interpolated_x)
        partial_gp_loss_x = partial(self._gradient_penalty_loss, averaged_samples=interpolated_x)
        partial_gp_loss_x.__name__ = 'gradient_penalty'
        self.critic_x_model = Model(inputs=[x, z], outputs=[valid_x, fake_x,
                                                            validity_interpolated_x])
        self.critic_x_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                          partial_gp_loss_x], optimizer=self.optimizer,
                                    loss_weights=[1, 1, 10])
        
        '''
        print("input: x_.shape: ", x_.shape)
        print("output: fake_x.shape: ", fake_x.shape)
        print("input: x.shape: ", x.shape)
        print("output: valid_x.shape: ", valid_x.shape)
        print("self.critic_x.summary(): ", self.critic_x.summary())
        print("self.critic_x_model.summary(): ", self.critic_x_model.summary())
        print()
        print()
        '''

        fake_z = self.critic_z(z_)
        valid_z = self.critic_z(z)
        interpolated_z = RandomWeightedAverage()([z, z_])
        validity_interpolated_z = self.critic_z(interpolated_z)
        partial_gp_loss_z = partial(self._gradient_penalty_loss, averaged_samples=interpolated_z)
        partial_gp_loss_z.__name__ = 'gradient_penalty'
        self.critic_z_model = Model(inputs=[x, z], outputs=[valid_z, fake_z,
                                                            validity_interpolated_z])
        self.critic_z_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                          partial_gp_loss_z], optimizer=self.optimizer,
                                    loss_weights=[1, 1, 10])

        '''
        print("input: z_.shape: ", z_.shape)
        print("output: fake_z.shape: ", fake_z.shape)
        print("input: z.shape: ", z.shape)
        print("output: valid_z.shape: ", valid_z.shape)
        print("self.critic_z.summary(): ", self.critic_z.summary())
        print("self.critic_z_model.summary(): ", self.critic_z_model.summary())
        print()
        print()
        '''

        self.critic_x.trainable = False
        self.critic_z.trainable = False
        self.generator.trainable = True
        self.encoder.trainable = True

        z_gen = Input(shape=(self.latent_dim, 1))
        x_gen_ = self.generator(z_gen)
        x_gen = Input(shape=self.shape)
        z_gen_ = self.encoder(x_gen)
        x_gen_rec = self.generator(z_gen_)
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        self.encoder_generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        self.encoder_generator_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                                   'mse'], optimizer=self.optimizer,
                                             loss_weights=[1, 1, 10])

        '''
        print("input: x.shape: ", x.shape)
        print("output: z_.shape: ", z_.shape)
        print("self.encoder.summary(): ", self.encoder.summary())
        print("input: z.shape: ", z.shape)
        print("output: x_.shape: ", x_.shape)
        print("self.generator.summary(): ", self.generator.summary())
        print("self.encoder_generator_model.summary(): ", self.encoder_generator_model.summary())
        print()
        print()
        
        print("self.encoder_input_shape: ", self.encoder_input_shape)
        print("self.generator_input_shape: ", self.generator_input_shape)
        print("self.critic_x_input_shape: ", self.critic_x_input_shape)
        print("self.critic_z_input_shape: ", self.critic_z_input_shape)
        '''

    def _fit(self, X):
        fake = np.ones((self.batch_size, 1))
        valid = -np.ones((self.batch_size, 1))
        delta = np.ones((self.batch_size, 1))
        

        X_ = np.copy(X)
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(X_)
            epoch_g_loss = []
            epoch_cx_loss = []
            epoch_cz_loss = []

            minibatches_size = self.batch_size * self.iterations_critic
            num_minibatches = int(X_.shape[0] // minibatches_size)

            for i in range(num_minibatches):
                minibatch = X_[i * minibatches_size: (i + 1) * minibatches_size]

                for j in range(self.iterations_critic):
                    x = minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    z = np.random.normal(size=(self.batch_size, self.latent_dim, 1))
                    epoch_cx_loss.append(
                        self.critic_x_model.train_on_batch([x, z], [valid, fake, delta]))
                    epoch_cz_loss.append(
                        self.critic_z_model.train_on_batch([x, z], [valid, fake, delta]))

                epoch_g_loss.append(
                    self.encoder_generator_model.train_on_batch([x, z], [valid, valid, x]))

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            g_loss = np.mean(np.array(epoch_g_loss), axis=0)
            print('Epoch: {}/{}, [Dx loss: {}] [Dz loss: {}] [G loss: {}]'.format(
                epoch, self.epochs, cx_loss, cz_loss, g_loss))
        
            self.total_cx_loss.append(cx_loss)
            self.total_cz_loss.append(cz_loss)
            self.total_g_loss.append(g_loss)
            

    def fit(self, X, **kwargs):
        """Fit the TadGAN.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
        """
        self._build_tadgan(**kwargs)
        #X = X.reshape((-1, self.shape[0], 1))              COMMEMNTED THIS OUT
        self._fit(X)
        
    def save_model(self, epoch):
        self.critic_x_model.save("criticx_"+epoch)
        self.critic_z_model.save("criticz"+epoch)
        self.generator.save("generator"+epoch)
        self.encoder.save("encoder"+epoch)
        self.critic_x.save("critic_x"+epoch)
        self.critic_z.save("critic_z"+epoch)
        self.encoder_generator_model.save("encoder_generator_"+epoch)
        
    def load_model(self, epoch):
        
        self.critic_x_model = keras.models.load_model("criticx_"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)
        self.critic_z_model = keras.models.load_model("criticz"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)
        self.generator = keras.models.load_model("generator"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)
        self.encoder = keras.models.load_model("encoder"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)
        self.critic_x = keras.models.load_model("critic_x"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)
        self.critic_z = keras.models.load_model("critic_z"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)
        self.encoder_generator_model = keras.models.load_model("encoder_generator_"+epoch, custom_objects={'RandomWeightedAverage':RandomWeightedAverage, 'self._wasserstein_loss':self._wasserstein_loss}, compile=False)

    def predict(self, X):
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the critic scores for each input sequence.
        """

        #X = X.reshape((-1, self.shape[0], 1))
        print(X.shape)
        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(X)

        return y_hat, critic


def _compute_critic_score(critics, smooth_window):
    """Compute an array of anomaly scores.

    Args:
        critics (ndarray):
            Critic values.
        smooth_window (int):
            Smooth window that will be applied to compute smooth errors.

    Returns:
        ndarray:
            Array of anomaly scores.
    """
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)

    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return z_scores


def score_anomalies(y, y_hat, critic, index, score_window=10, critic_smooth_window=None,
                    error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult",
                    lambda_rec=0.5):
    """Compute an array of anomaly scores.

    Anomaly scores are calculated using a combination of reconstruction error and critic score.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            time index for each y (start position of the window)
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        critic_smooth_window (int):
            Optional. Size of window over which smoothing is applied to critic.
            If not given, 200 is used.
        error_smooth_window (int):
            Optional. Size of window over which smoothing is applied to error.
            If not given, 200 is used.
        smooth (bool):
            Optional. Indicates whether errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'point' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of anomaly scores.
    """

    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)

    step_size = 1  # expected to be 1

    true_index = index  # no offset

    '''
    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)
    '''
    
    true = []
    for item in y:
        true.append(item[0])

    for item in y[-1][1:]:
        true.append(item)

    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())

    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        critic_intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            critic_intermediate.append(critic_extended[i - j, j])

        if len(critic_intermediate) > 1:
            discr_intermediate = np.asarray(critic_intermediate)
            try:
                critic_kde_max.append(discr_intermediate[np.argmax(
                    stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
            except np.linalg.LinAlgError:
                critic_kde_max.append(np.median(discr_intermediate))
        else:
            critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    # Compute critic scores
    critic_scores = _compute_critic_score(critic_kde_max, critic_smooth_window)
    
#     print("y.shape, y_hat.shape, step_size, score_window, error_smooth_window, smooth, rec_error_type: " )
#     print(y.shape, y_hat.shape, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    # Compute reconstruction scores
    rec_scores, predictions = reconstruction_errors(
        y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    rec_scores = stats.zscore(rec_scores)
    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1
    
#     print("critic_scores.shape, rec_scores.shape")
#     print(critic_scores.shape, rec_scores.shape)

    # Combine the two scores
    if comb == "mult":
        final_scores = np.multiply(critic_scores, rec_scores)

    elif comb == "sum":
        final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)

    elif comb == "rec":
        final_scores = rec_scores

    else:
        raise ValueError(
            'Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))

    #true = [[t] for t in true]
    
#     print("final_scores.shape, true_index.shape, true.shape, predictions.shape")
#     print(np.array(final_scores).shape, np.array(true_index).shape, np.array(true).shape, np.array(predictions).shape)
    return final_scores, true_index, true, predictions