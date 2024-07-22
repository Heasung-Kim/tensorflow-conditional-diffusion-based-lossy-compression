import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.mixed_precision import global_policy as mixed_precision
import time

def batch_psnr(imgs1, imgs2):
    batch_mse = tf.reduce_mean(tf.square(imgs1 - imgs2), axis=(1, 2, 3))
    batch_psnr = 20 * tf.math.log(1.0 / tf.sqrt(batch_mse)) / tf.math.log(10.0)
    return tf.reduce_mean(batch_psnr)

def batch_nmse(compressed, batch):
    x_test_real = tf.reshape(batch[:, 0, :, :], [tf.shape(batch)[0], -1])
    x_test_imag = tf.reshape(batch[:, 1, :, :], [tf.shape(batch)[0], -1])
    x_test_C = tf.complex(x_test_real - 0.5, x_test_imag - 0.5)

    x_hat_real = tf.reshape(compressed[:, 0, :, :], [tf.shape(compressed)[0], -1])
    x_hat_imag = tf.reshape(compressed[:, 1, :, :], [tf.shape(compressed)[0], -1])
    x_hat_C = tf.complex(x_hat_real - 0.5, x_hat_imag - 0.5)

    # Compute power and MSE
    power = tf.reduce_sum(tf.abs(x_test_C) ** 2, axis=1)
    mse = tf.reduce_sum(tf.abs(x_test_C - x_hat_C) ** 2, axis=1)

    # Compute NMSE
    nmse = 10 * tf.math.log(tf.reduce_mean(mse / power)) / tf.math.log(10.0)

    return nmse


class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl,
        data_generator,
        scheduler_function,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        ema_decay=0.999,
        ema_update_interval=10,
        ema_step_start=100,
        use_mixed_precision=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        if train_dl is not None and val_dl is not None:
            self.train_dl = iter(train_dl.repeat())
            self.val_dl = iter(val_dl)
        else:
            self.train_dl = None
            self.val_dl = None
            self.data_generator = data_generator

        self.init_train_lr = train_lr
        if optimizer == "adam":
            self.opt = Adam(learning_rate=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(learning_rate=train_lr)
        self.scheduler = scheduler_function
        self.ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        self.ema_update_interval = ema_update_interval
        self.ema_step_start = ema_step_start
        self.scaler = mixed_precision.LossScaleOptimizer(self.opt) if use_mixed_precision else None

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.writer = tf.summary.create_file_writer(tensorboard_dir)



    def save(self):
        data = {
            "step": self.step,
            "model": self.model.get_weights(),
            "ema": {var.name: var.numpy() for var in self.model.trainable_variables}
        }
        idx = (self.step // self.save_and_sample_every) % 3
        np.save(os.path.join(self.results_folder, f"{self.model_name}_{idx}.npy"), data)

    def load(self, idx=0, load_step=True):
        data = np.load(os.path.join(self.results_folder, f"{self.model_name}_{idx}.npy"), allow_pickle=True).item()
        self.model.set_weights(data["model"])
        #ema_restore = {self.model.get_layer(name).trainable_variables[0]: value for name, value in data["ema"].items()}
        #self.ema.apply(ema_restore)
        if load_step:
            self.step = data["step"]

    @tf.function
    def update(self, data):
        with tf.GradientTape() as tape:
            loss, aloss = self.model(data * 2.0 - 1.0, training=True)

            total_loss = loss + aloss
            if self.scaler:
                scaled_loss = self.scaler.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        if self.scaler:
            scaled_gradients = self.scaler.get_unscaled_gradients(gradients)
            gradients = [tf.clip_by_norm(grad, 1.0) for grad in scaled_gradients]
            self.scaler.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.scaler.update()
        else:
            gradients = [tf.clip_by_norm(grad, 1.0) for grad in gradients]
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, aloss

    def train(self):
        while self.step < self.train_num_steps:
            self.opt.learning_rate = self.init_train_lr * self.scheduler(self.step-self.scheduler_checkpoint_step) if self.step >= self.scheduler_checkpoint_step else self.opt.learning_rate

            if self.train_dl is not None:
                data = next(self.train_dl)
            else:
                data = self.data_generator.get_batch()
            data = data[..., 0]
            #forward_start_time = time.time()
            loss, aloss = self.update(data)

            #print("gradient_computation_time:{}".format(time.time() - forward_start_time))

            with self.writer.as_default():
                tf.summary.scalar("loss", loss, step=self.step)

            #if self.step >= self.ema_step_start and self.step % self.ema_update_interval == 0:
            #    self.ema.apply(self.model.trainable_variables)

            if self.step % self.save_and_sample_every == 0:

                if self.val_dl is not None:
                    batch = next(self.val_dl)
                else:
                    batch = self.data_generator.get_batch()
                batch = batch[..., 0]

                i=0
                self.ema.apply(self.model.trainable_variables)

                compressed, bpp = self.model.predict(batch * 2.0 - 1.0, self.sample_steps)
                compressed = (compressed + 1.0) * 0.5

                compressed = tf.transpose(compressed, perm=[0,2,3,1])
                batch = tf.transpose(batch, perm=[0,2,3,1])

                batch_nmse_val = batch_nmse(tf.clip_by_value(compressed, 0.0, 1.0), batch)
                step_tmp = self.step // self.save_and_sample_every
                print("step:{}, nmse:{}, loss:{}/aloss:{}".format(step_tmp, batch_nmse_val, loss, aloss))
                with self.writer.as_default():
                    tf.summary.scalar(f"bpp/num{i}", bpp, step=self.step // self.save_and_sample_every)
                    tf.summary.scalar(f"psnr/num{i}", batch_psnr(tf.clip_by_value(compressed, 0.0, 1.0), batch), step=self.step // self.save_and_sample_every)
                    tf.summary.scalar(f"nmse/num{i}", batch_nmse_val, step=self.step // self.save_and_sample_every)
                    tf.summary.image(f"compressed/num{i}", tf.clip_by_value(compressed, 0.0, 1.0), step=self.step // self.save_and_sample_every)
                    tf.summary.image(f"original/num{i}", batch, step=self.step // self.save_and_sample_every)
                self.save()

            self.step += 1
        self.save()
        print("training completed")