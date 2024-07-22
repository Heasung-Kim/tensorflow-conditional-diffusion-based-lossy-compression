import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

@tf.function
def extract(a, t, batch_size, x_shape):
    b = batch_size #tf.shape(t)[0]
    out = tf.gather(a, t, axis=-1)
    return tf.reshape(out, (b, *([1] * (len(x_shape) - 1))))

def extract_tensor(a, t, place_holder=None):
    return tf.gather_nd(a, tf.stack([t, tf.range(tf.shape(t)[0])], axis=1))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: tf.tile(tf.random.normal((1, *shape[1:]), device=device), [shape[0], *([1] * (len(shape) - 1))])
    noise = lambda: tf.random.normal(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def round_w_offset(input, loc):
    diff = STERound(input - loc)
    return diff + loc

def noise(x, scale):
    return x + scale * (tf.random.uniform(tf.shape(x)) - 0.5)

def quantize(x, mode='noise', offset=None):
    if mode == 'noise':
        return noise(x, 1)
    elif mode == 'round':
        return STERound(x)
    elif mode == 'dequantize':
        return round_w_offset(x, offset)
    else:
        raise NotImplementedError


@tf.custom_gradient
def STERound(x):
    rounded = tf.round(x)

    def grad(dy):
        return dy  # Straight-through gradient

    return rounded, grad

@tf.custom_gradient
def lower_bound(inputs, bound):
    def grad(dy):
        pass_through_1 = inputs >= bound
        pass_through_2 = dy < 0
        pass_through = pass_through_1 | pass_through_2
        return tf.cast(pass_through, dy.dtype) * dy, None
    b = tf.ones_like(inputs) * bound
    return tf.maximum(inputs, b), grad

class LowerBound(tf.keras.layers.Layer):
    def call(self, inputs, bound):
        return lower_bound(inputs, bound)

@tf.custom_gradient
def upper_bound(inputs, bound):
    def grad(dy):
        pass_through_1 = inputs <= bound
        pass_through_2 = dy > 0
        pass_through = pass_through_1 | pass_through_2
        return tf.cast(pass_through, dy.dtype) * dy, None
    b = tf.ones_like(inputs) * bound
    return tf.minimum(inputs, b), grad

class UpperBound(tf.keras.layers.Layer):
    def call(self, inputs, bound):
        return upper_bound(inputs, bound)

class NormalDistribution:
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return tf.stop_gradient(self.loc)

    @tf.function
    def std_cdf(self, inputs):
        half = 0.5
        const = -(2 ** -0.5)
        return half * tf.math.erfc(const * inputs)

    @tf.function
    def sample(self):
        return self.scale * tf.random.normal(tf.shape(self.scale)) + self.loc

    @tf.function
    def likelihood(self, x, min=1e-9):
        x = tf.abs(x - self.loc)
        upper = self.std_cdf((0.5 - x) / self.scale)
        lower = self.std_cdf((-0.5 - x) / self.scale)
        return lower_bound(upper - lower, min)

    @tf.function
    def scaled_likelihood(self, x, s=1, min=1e-9):
        x = tf.abs(x - self.loc)
        s = s * 0.5
        upper = self.std_cdf((s - x) / self.scale)
        lower = self.std_cdf((-s - x) / self.scale)
        return lower_bound(upper - lower, min)



class EMA:
    def __init__(self, model, decay, updates_per_epoch, warmup):
        self.ema_model = tf.keras.models.clone_model(model)
        self.ema_model.set_weights(model.get_weights())
        self.decay = decay
        self.num_updates = 0
        self.updates_per_epoch = updates_per_epoch
        self.warmup = warmup

    def update(self, model):
        self.num_updates += 1
        if self.num_updates < self.warmup:
            decay = 0
        else:
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        new_weights = model.get_weights()
        ema_weights = self.ema_model.get_weights()

        for i in range(len(new_weights)):
            ema_weights[i] = decay * ema_weights[i] + (1 - decay) * new_weights[i]

        self.ema_model.set_weights(ema_weights)

    def get_weights(self):
        return self.ema_model.get_weights()

    def set_weights(self, weights):
        self.ema_model.set_weights(weights)
