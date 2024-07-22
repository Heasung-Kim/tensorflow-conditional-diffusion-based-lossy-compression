import time

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, losses
from functools import partial
from tqdm import tqdm
from .utils import cosine_beta_schedule, linear_beta_schedule, extract

class GaussianDiffusion(tf.keras.Model):
    def __init__(self, denoise_fn, context_fn, input_img_shape, batch_size, ae_fn=None, num_timesteps=1000,
                 loss_type="l1", lagrangian=1e-3, pred_mode="noise",
                 var_schedule="linear", aux_loss_weight=0, aux_loss_type="l1",
                 use_loss_weight=False, loss_weight_min=5,
                 use_aux_loss_weight_schedule=False):
        super(GaussianDiffusion, self).__init__()
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.input_img_shape = input_img_shape
        self.batch_size = batch_size
        self.ae_fn = ae_fn
        self.loss_type = loss_type
        self.lagrangian_beta = lagrangian
        self.var_schedule = var_schedule
        self.sample_steps = None
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_type = aux_loss_type
        self.use_aux_loss_weight_schedule = use_aux_loss_weight_schedule
        assert pred_mode in ["noise", "x", "v"]
        self.pred_mode = pred_mode
        self.use_loss_weight = use_loss_weight
        self.loss_weight_min = float(loss_weight_min)

        if var_schedule == "cosine":
            train_betas = cosine_beta_schedule(num_timesteps)
        elif var_schedule == "linear":
            train_betas = linear_beta_schedule(num_timesteps)

        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)

        self.num_timesteps = int(train_betas.shape[0])

        self.train_snr = tf.convert_to_tensor(train_alphas_cumprod / (1 - train_alphas_cumprod), dtype=tf.float32)
        self.train_betas = tf.convert_to_tensor(train_betas, dtype=tf.float32)
        self.train_alphas_cumprod = tf.convert_to_tensor(train_alphas_cumprod, dtype=tf.float32)
        self.train_sqrt_alphas_cumprod = tf.convert_to_tensor(np.sqrt(train_alphas_cumprod), dtype=tf.float32)
        self.train_sqrt_one_minus_alphas_cumprod = tf.convert_to_tensor(np.sqrt(1.0 - train_alphas_cumprod), dtype=tf.float32)
        self.train_sqrt_recip_alphas_cumprod = tf.convert_to_tensor(np.sqrt(1.0 / train_alphas_cumprod), dtype=tf.float32)
        self.train_sqrt_recipm1_alphas_cumprod = tf.convert_to_tensor(np.sqrt(1.0 / train_alphas_cumprod - 1), dtype=tf.float32)

        self.training = True

        self.x_start_shape = (self.batch_size, 2, 32, 32)
        self.noise_shape = (self.batch_size, 2, 32, 32)
        self.img_batch_shape = (self.batch_size, 2, 32, 32)

        #self.build(input_shape=self.img_batch_shape)
        
    def parameters(self, skip_keywords=["loss_fn_vgg", "ae_fn"], recurse=True):
        for var in self.trainable_variables:
            use = True
            for keyword in skip_keywords:
                if keyword in var.name:
                    use = False
                    break
            if use:
                yield var

    @tf.function
    def get_extra_loss(self):
        return self.context_fn.get_extra_loss()

    #@tf.function
    def set_sample_schedule(self, sample_steps):
        self.sample_steps = sample_steps
        if sample_steps != 1:
            indice = tf.cast(tf.linspace(0, self.num_timesteps - 1, sample_steps), tf.int32)
        else:
            indice = tf.convert_to_tensor([self.num_timesteps - 1], dtype=tf.int32)
        self.alphas_cumprod = tf.gather(self.train_alphas_cumprod, indice)
        self.snr = tf.gather(self.train_snr, indice)
        self.index = tf.gather(tf.range(self.num_timesteps, dtype=tf.int32), indice)
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = tf.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = tf.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = tf.sqrt(1.0 / self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sigma = self.sqrt_one_minus_alphas_cumprod_prev / self.sqrt_one_minus_alphas_cumprod * tf.sqrt(1.0 - self.alphas_cumprod / self.alphas_cumprod_prev)

    def predict_noise_from_start(self, x_t, t, x0):
        return (tf.gather(self.sqrt_recip_alphas_cumprod, t) * x_t - x0) / tf.gather(self.sqrt_recipm1_alphas_cumprod, t)

    def predict_v(self, x_start, t, noise):
        if self.training:
            return (extract(self.train_sqrt_alphas_cumprod, t, self.x_start_shape) * noise -
                    extract(self.train_sqrt_one_minus_alphas_cumprod, t, self.x_start_shape) * x_start)
        else:
            return (extract(self.sqrt_alphas_cumprod, t, self.x_start_shape) * noise -
                    extract(self.sqrt_one_minus_alphas_cumprod, t, self.x_start_shape) * x_start)

    def predict_start_from_v(self, x_t, t, v):
        if self.training:
            return (extract(self.train_sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)
        else:
            return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def predict_start_from_noise(self, x_t, t, noise):
        if self.training:
            return (extract(self.train_sqrt_recip_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_t -
                    extract(self.train_sqrt_recipm1_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * noise)
        else:
            return (extract(self.sqrt_recip_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * noise)

    @tf.function
    def ddim(self, x, t, context, clip_denoised, eta=0):
        if self.denoise_fn.embd_type == "01":
            #asdf = tf.gather(self.index,t)
            tt = tf.expand_dims(tf.cast(tf.gather(self.index, t), tf.float32), -1) / self.num_timesteps
            fx = self.denoise_fn(x, tt, context)

        else:
            fx = self.denoise_fn(x, tf.gather(self.index,t), context)
        if self.pred_mode == "noise":
            x_recon = self.predict_start_from_noise(x, t, fx)
        elif self.pred_mode == "x":
            x_recon = fx
        elif self.pred_mode == "v":
            x_recon = self.predict_start_from_v(x, t, fx)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, -1.0, 1.0)
        noise = fx if self.pred_mode == "noise" else self.predict_noise_from_start(x, t, x_recon)
        x_next = (
                extract(self.sqrt_alphas_cumprod_prev, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_recon +
                  tf.sqrt(
                      tf.clip_by_value(extract(self.one_minus_alphas_cumprod_prev, t, batch_size=self.batch_size, x_shape=self.x_start_shape) -
                           (eta * extract(self.sigma, t, batch_size=self.batch_size, x_shape=self.x_start_shape)) ** 2,clip_value_min=0, clip_value_max=1.0)) * noise +
                  eta * extract(self.sigma, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * tf.random.normal(self.noise_shape))
        return x_next

    @tf.function
    def p_sample(self, x, t, context, clip_denoised, eta=0):
        return self.ddim(x, t, context, clip_denoised, eta)

    @tf.function
    def p_sample_loop(self, shape, context, clip_denoised=False, init=None, eta=0):
        b = shape[0]
        img = tf.zeros(shape) if init is None else init
        for i in tqdm(reversed(range(self.sample_steps)), desc="sampling loop time step", total=self.sample_steps):
            time = tf.fill([b], i)
            img = self.p_sample(img, time, context, clip_denoised, eta)
        return img

    #@tf.function
    def predict(self, images, sample_steps=None, bpp_return_mean=True, init=None, eta=0):
        self.training = False
        compress_start_time = time.time()
        context_dict = self.context_fn(images)
        print("context_fn {}sec".format(time.time() - compress_start_time))
        compress_start_time = time.time()
        self.set_sample_schedule(self.num_timesteps if sample_steps is None else sample_steps)
        print("set_sample_schedule takes {}sec".format(time.time() - compress_start_time))

        if self.ae_fn is None:
            compress_start_time = time.time()
            compressed_image = self.p_sample_loop(self.img_batch_shape, context_dict['output'], clip_denoised=True, init=init,
                                                  eta=eta)
            bpp = tf.reduce_mean(context_dict["bpp"]) if bpp_return_mean else context_dict["bpp"]
            self.training = True
            print("encode/decode takes {} sec".format(time.time() - compress_start_time))
            return compressed_image, bpp
        else:
            z = self.ae_fn.encode(images).mode()
            dec_z = self.p_sample_loop(z.shape, context_dict['output'], clip_denoised=False, init=init, eta=eta)
            decoded_image = self.ae_fn.decode(dec_z)
            bpp = tf.reduce_mean(context_dict["bpp"]) if bpp_return_mean else context_dict["bpp"]
            self.training = True
            return decoded_image, bpp

    @tf.function
    def q_sample(self, x_start, t, noise):
        sample = (extract(self.train_sqrt_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_start
                  + extract(self.train_sqrt_one_minus_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * noise)
        return sample

    @tf.function
    def p_losses(self, x_start, context_dict, t, aux_img=None):
        noise = tf.random.normal(shape=self.x_start_shape)
        x_noisy = self.q_sample(x_start, t, noise)

        if self.denoise_fn.embd_type == "01":
            #tt = tf.expand_dims(tf.cast(t, tf.float32) , -1) / self.num_timesteps
            fx = self.denoise_fn(x_noisy, tf.expand_dims(tf.cast(t, tf.float32) , -1) / self.num_timesteps, context_dict['output'])
        else:
            fx = self.denoise_fn(x_noisy, t, context_dict['output'])

        if self.pred_mode == "noise":
            weight = tf.ones(1)
            if self.loss_type == "l2":
                err = tf.reduce_mean(tf.square(noise - fx), axis=[1, 2, 3])
                err = tf.reduce_mean(err * weight)
            else:
                raise NotImplementedError()
        elif self.pred_mode == "x":
            if self.use_loss_weight:
                weight = tf.clip_by_value(self.train_snr[t], 0,
                                          self.loss_weight_min) if self.loss_weight_min > 0 else tf.clip_by_value(
                    self.train_snr[t], self.loss_weight_min, 1e8)
            else:
                weight = tf.ones(1)
            if self.loss_type == "l1":
                err = tf.reduce_mean(tf.abs(x_start - fx), axis=[1, 2, 3])
                err = tf.reduce_mean(err * tf.sqrt(weight))
            elif self.loss_type == "l2":
                err = tf.reduce_mean(tf.square(x_start - fx), axis=[1, 2, 3])
                err = tf.reduce_mean(err * weight)
            else:
                raise NotImplementedError()
        elif self.pred_mode == "v":
            if self.use_loss_weight:
                weight = tf.clip_by_value(self.train_snr[t], 0, self.loss_weight_min) / (
                            self.train_snr[t] + 1) if self.loss_weight_min > 0 else self.train_snr[t] / (
                            self.train_snr[t] + 1)
            else:
                weight = self.train_snr[t] / (self.train_snr[t] + 1)
            v = self.predict_v(x_start, t, noise)
            if self.loss_type == "l1":
                err = tf.reduce_mean(tf.abs(fx - v), axis=[1, 2, 3])
                err = tf.reduce_mean(err * tf.sqrt(weight))
            elif self.loss_type == "l2":
                err = tf.reduce_mean(tf.square(fx - v), axis=[1, 2, 3])
                err = tf.reduce_mean(err * weight)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        aux_err = 0

        if self.aux_loss_weight > 0:
            if self.pred_mode == "noise":
                pred_x0 = self.predict_start_from_noise(x_noisy, t, fx)
            elif self.pred_mode == "x":
                pred_x0 = fx
            elif self.pred_mode == "v":
                pred_x0 = self.predict_start_from_v(x_noisy, t, fx)
            if self.ae_fn is not None:
                pred_x0 = self.ae_fn.decode(pred_x0)

            if self.use_aux_loss_weight_schedule:
                weight = tf.clip_by_value(self.train_snr[t], 0,
                                          self.loss_weight_min) if self.loss_weight_min > 0 else tf.clip_by_value(
                    self.train_snr[t], self.loss_weight_min, 1e8)
            else:
                weight = tf.ones(1)

            if self.aux_loss_type == "l1":
                aux_err = tf.reduce_mean(tf.abs(aux_img - pred_x0), axis=[1, 2, 3]) * tf.sqrt(weight)
            elif self.aux_loss_type == "l2":
                aux_err = tf.reduce_mean(tf.square(aux_img - pred_x0), axis=[1, 2, 3]) * weight
            elif self.aux_loss_type == "lpips":
                aux_err = []
                for i in range(aux_img.shape[0]):
                    aux_err.append(self.loss_fn_vgg(aux_img[i:i + 1], pred_x0[i:i + 1]))
                aux_err = tf.stack(aux_err, axis=0)
                aux_err = tf.reduce_mean(aux_err * weight)
            else:
                raise NotImplementedError()

            loss = (self.lagrangian_beta * tf.reduce_mean(context_dict["bpp"]) +
                    err * (1 - self.aux_loss_weight) + aux_err * self.aux_loss_weight)
        else:
            loss = self.lagrangian_beta * tf.reduce_mean(context_dict["bpp"]) + err

        return loss

    @tf.function
    def call(self, images):
        # B, C, H, W = self.img_batch_shape
        B = self.batch_size #tf.shape(images)[0]
        t = tf.random.uniform([B], minval=0, maxval=self.num_timesteps, dtype=tf.int32)
        output_dict = self.context_fn(images, training=True)
        if self.ae_fn is not None:
            z = self.ae_fn.encode(images).mode()
            loss = self.p_losses(z, output_dict, t, aux_img=images)
        else:
            loss = self.p_losses(images, output_dict, t, aux_img=images)
        return loss, self.get_extra_loss()

