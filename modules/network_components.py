import tensorflow as tf
import math
import numpy as np
from einops import rearrange
from .utils import exists, lower_bound

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
class Residual(tf.keras.layers.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def call(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(x, -1) * tf.expand_dims(emb, 0)
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class Upsample(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out=None):
        super(Upsample, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        #self.padding = ZeroPadding2D(padding=1, data_format="channels_first")
        self.conv = Conv2DTranspose(dim_out, kernel_size=4, strides=2, padding='same', data_format="channels_first")

    def call(self, x):
        #x = self.padding(x)
        x = self.conv(x)
        return x


class Downsample(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out=None):
        super(Downsample, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.padding = ZeroPadding2D(padding=1, data_format="channels_first")
        self.conv = Conv2D(dim_out, kernel_size=3, strides=2, padding='valid', data_format="channels_first")

    def call(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return x


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = self.add_weight(shape=(1, dim, 1, 1), initializer='ones', trainable=True)
        self.b = self.add_weight(shape=(1, dim, 1, 1), initializer='zeros', trainable=True)

    @tf.function
    def call(self, x):
        var = tf.math.reduce_variance(x, axis=1, keepdims=True)
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps) * self.g + self.b


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        #self.norm = LayerNorm(dim)
        self.norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], epsilon=1e-5)

    def call(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(tf.keras.layers.Layer):
    def __init__(self, dim, dim_out, large_filter=False):
        super(Block, self).__init__()

        self.padding = tf.keras.layers.ZeroPadding2D(padding=(3 if large_filter else 1),  data_format='channels_first')
        self.conv = tf.keras.layers.Conv2D(dim_out, kernel_size=(7 if large_filter else 3), padding='valid', data_format='channels_first')
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], epsilon=1e-5) # LayerNorm(dim_out) #
        self.relu_activation = tf.keras.layers.ReLU()
        # self.block = tf.keras.Sequential([
        #    tf.keras.layers.Conv2D(dim_out, 7 if large_filter else 3, padding=3 if large_filter else 1),
        #    LayerNorm(dim_out),
        #    tf.keras.layers.ReLU()
        # ])

    def call(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.relu_activation(x)
        return x


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False):
        super(ResnetBlock, self).__init__()
        self.mlp = (tf.keras.Sequential([
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dense(dim_out)
        ]) if exists(time_emb_dim) else None)

        self.block1 = Block(dim, dim_out, large_filter)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = (tf.keras.layers.Conv2D(dim_out, 1, data_format='channels_first') if dim != dim_out else tf.keras.layers.Identity())

    def call(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            y = self.mlp(time_emb)
            h = h + tf.expand_dims(tf.expand_dims(y, -1), -1)

        h = self.block2(h)
        residual_convolution = self.res_conv(x)
        return h + residual_convolution



class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=1, dim_head=None):
        super(LinearAttention, self).__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = tf.keras.layers.Conv2D(hidden_dim * 3, 1, use_bias=False, data_format="channels_first")
        self.to_out = tf.keras.layers.Conv2D(dim, 1, data_format="channels_first")


    @tf.function
    def call(self, x):
        b, c, h, w = x.shape
        qkv = tf.split(self.to_qkv(x), 3, axis=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y  -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        k = tf.nn.softmax(k, axis=-1)
        context = tf.einsum("b h d n, b h e n -> b h d e", k, v)

        out = tf.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class LinearAttentionasdf(tf.keras.layers.Layer):
    def __init__(self, dim, heads=1, dim_head=None):
        super(LinearAttention, self).__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = tf.keras.layers.Conv2D(hidden_dim * 3, 1, use_bias=False, data_format="channels_first")
        self.to_out = tf.keras.layers.Conv2D(dim, 1, data_format="channels_first")

        self.multiheadattention = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=hidden_dim)
    @tf.function
    def call(self, x):
        qkv = tf.split(self.to_qkv(x), 3, axis=1)
        out = self.multiheadattention(*qkv)
        return self.to_out(out)

class LearnedSinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = self.add_weight(shape=(half_dim,), initializer='random_normal', trainable=True)

    def call(self, x):
        x = tf.expand_dims(x, -1)
        freqs = x * tf.expand_dims(self.weights, 0) * 2 * math.pi
        fouriered = tf.concat([tf.sin(freqs), tf.cos(freqs)], axis=-1)
        return tf.concat([x, fouriered], axis=-1)


class ImprovedSinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim, is_random=False):
        super(ImprovedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = self.add_weight(shape=(half_dim,), initializer='random_normal', trainable=not is_random)

    def call(self, x):
        x = tf.expand_dims(x, -1)
        freqs = x * tf.expand_dims(self.weights, 0) * 2 * math.pi
        fouriered = tf.concat([tf.sin(freqs), tf.cos(freqs)], axis=-1)
        return tf.concat([x, fouriered], axis=-1)


class VBRCondition(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(VBRCondition, self).__init__()
        self.scale = tf.keras.layers.Conv2D(output_dim, 1, data_format="channels_first")
        self.shift = tf.keras.layers.Conv2D(output_dim, 1, data_format="channels_first")

    def call(self, input, cond):
        cond = tf.reshape(cond, (-1, 1, 1, 1))
        scale = self.scale(cond)
        shift = self.shift(cond)
        return input * scale + shift


class GDN(tf.keras.layers.Layer):
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=0.1, reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        beta = tf.sqrt(tf.ones(ch) + self.pedestal)
        self.beta = self.add_weight(name='beta', shape=(ch,), initializer=tf.constant_initializer(beta), trainable=True)

        eye = tf.eye(ch)
        g = self.gamma_init * eye + self.pedestal
        gamma = tf.sqrt(g)
        self.gamma = self.add_weight(name='gamma', shape=(ch, ch), initializer=tf.constant_initializer(gamma),
                                     trainable=True)

    def call(self, inputs):
        unfold = False
        if len(inputs.shape) == 5:
            unfold = True
            bs, ch, d, w, h = inputs.shape
            inputs = tf.reshape(inputs, (bs, ch, d * w, h))

        beta = tf.maximum(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        gamma = tf.maximum(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = tf.reshape(gamma, (gamma.shape[0], gamma.shape[1], 1, 1))

        norm_ = tf.nn.conv2d(inputs ** 2, gamma, strides=[1, 1, 1, 1], padding='same', data_format="channels_first") + beta
        norm_ = tf.sqrt(norm_)

        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = tf.reshape(outputs, (bs, ch, d, w, h))
        return outputs


class GDN1(GDN):
    def call(self, inputs):
        unfold = False
        if len(inputs.shape) == 5:
            unfold = True




class PriorFunction(tf.keras.layers.Layer):
    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True):
        super(PriorFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.add_weight(shape=(parallel_dims, 1, 1, in_features, out_features), trainable=True)
        if bias:
            self.bias = self.add_weight(shape=(parallel_dims, 1, 1, 1, out_features), trainable=True)
        else:
            self.bias = None

    @tf.function
    def call(self, input, detach=False):
        if detach:
            weight = tf.nn.softplus(tf.stop_gradient(self.weight))
            bias = tf.stop_gradient(self.bias) if self.bias is not None else None
        else:
            weight = tf.nn.softplus(self.weight)
            bias = self.bias
        return tf.matmul(input, weight) + bias

    def get_config(self):
        return {'in_features': self.in_features, 'out_features': self.out_features, 'bias': self.bias is not None}


class FlexiblePrior(tf.keras.layers.Layer):
    def __init__(self, channels=256, dims=[3, 3, 3], init_scale=10.):
        super(FlexiblePrior, self).__init__()
        dims = [1] + dims + [1]
        self.chain_len = len(dims) - 1
        scale = init_scale**(1 / self.chain_len)
        h_b = []
        for i in range(self.chain_len):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            h_b.append(PriorFunction(channels, dims[i], dims[i + 1], init))
        self.affine = h_b
        self.a = [self.add_weight(shape=(channels, 1, 1, 1, dims[i + 1]), initializer='zeros', trainable=True) for i in range(self.chain_len - 1)]

        self._medians = self.add_weight(shape=(1, channels, 1, 1), initializer='zeros', trainable=True)

    @property
    def medians(self):
        return tf.stop_gradient(self._medians)

    @tf.function
    def cdf(self, x, logits=True, detach=False):
        x = tf.expand_dims(tf.transpose(x, [1, 0, 2, 3]), -1)  # C, N, H, W, 1
        for i in range(self.chain_len - 1):
            x = self.affine[i](x, detach)
            if detach is True:
                a_temp = tf.stop_gradient(self.a[i])
            else:
                a_temp = self.a[i]
            x = x + tf.tanh(a_temp) * tf.tanh(x)
        if logits:
            return tf.squeeze(tf.transpose(self.affine[-1](x, detach), [1, 0, 2, 3, 4]), -1)
        return tf.sigmoid(tf.squeeze(tf.transpose(self.affine[-1](x, detach), [1, 0, 2, 3, 4]), -1))

    @tf.function
    def pdf(self, x):
        cdf = self.cdf(x, False)
        jac = tf.ones_like(cdf)
        with tf.GradientTape() as tape:
            tape.watch(x)
            cdf_val = self.cdf(x, False)
        pdf = tape.gradient(cdf_val, x, output_gradients=jac)
        return pdf

    @tf.function
    def get_extraloss(self):
        target = 0.
        logits = self.cdf(self._medians, detach=True)
        extra_loss = tf.reduce_sum(tf.abs(logits - target))
        return extra_loss

    @tf.function
    def likelihood(self, x, min=1e-9):
        lower = self.cdf(x - 0.5, True)
        upper = self.cdf(x + 0.5, True)
        sign = -tf.stop_gradient(tf.sign(lower + upper))
        upper = tf.sigmoid(upper * sign)
        lower = tf.sigmoid(lower * sign)
        return lower_bound(tf.abs(upper - lower), min)

    @tf.function
    def icdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9):
        if method == 'bisection':
            init_interval = [-1, 1]
            left_endpoints = tf.ones_like(xi) * init_interval[0]
            right_endpoints = tf.ones_like(xi) * init_interval[1]

            def f(z):
                return self.cdf(z, logits=False, detach=True) - xi

            while True:
                if tf.reduce_all(f(left_endpoints) < 0):
                    break
                else:
                    left_endpoints *= 2
            while True:
                if tf.reduce_all(f(right_endpoints) > 0):
                    break
                else:
                    right_endpoints *= 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = tf.logical_not(pos)
                neg = mid_vals < 0
                non_neg = tf.logical_not(neg)
                left_endpoints = left_endpoints * tf.cast(non_neg, tf.float32) + mid_pts * tf.cast(neg, tf.float32)
                right_endpoints = right_endpoints * tf.cast(non_pos, tf.float32) + mid_pts * tf.cast(pos, tf.float32)
                if tf.reduce_all(tf.logical_and(non_pos, non_neg)) or tf.reduce_min(right_endpoints - left_endpoints) <= tol:
                    print(f'bisection terminated after {i} iterations')
                    break

            return mid_pts
        else:
            raise NotImplementedError

    @tf.function
    def sample(self, img, shape):
        uni = tf.random.uniform(shape, minval=0, maxval=1, dtype=img.dtype)
        return self.icdf(uni)
