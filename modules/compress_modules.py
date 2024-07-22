import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, ZeroPadding2D
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution, lower_bound


class Compressor(tf.keras.Model):
    def __init__(
        self,
            batch_size,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
    ):
        super(Compressor, self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])

        self.batch_size = batch_size

        self.x_start_shape = (self.batch_size, 2, 32, 32)
        self.noise_shape = (self.batch_size, 2, 32, 32)
        self.img_batch_shape = (self.batch_size, 2, 32, 32)


        self.build_network()

    @tf.function
    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = []
        self.dec = []
        self.hyper_enc = []
        self.hyper_dec = []

    def deprecated_encode(self, input):
        for resnet, down in self.enc:
            input = resnet(input)
            input = down(input)
        latent = input
        for padding, conv, act in self.hyper_enc:
            input = padding(input)
            input = conv(input)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for padding, deconv, act in self.hyper_dec:
            input = padding(input)
            input = deconv(input)
            input = act(input)
        mean, scale = tf.split(input, 2, axis=1)
        latent_distribution = NormalDistribution(mean, tf.clip_by_value(scale, 0.1, 10.0))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    @tf.function
    def encode(self, input):
        for resnet, down in self.enc:
            input = resnet(input)
            input = down(input)
        latent = input
        for padding, conv, act in self.hyper_enc:
            input = padding(input)
            input = conv(input)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for padding, deconv, act in self.hyper_dec:
            input = padding(input)
            input = deconv(input)
            input = act(input)
        mean, scale = tf.split(input, 2, axis=1)
        #latent_distribution = NormalDistribution(mean, tf.clip_by_value(scale, 0.1, 10.0))
        q_latent = quantize(latent, "dequantize", tf.stop_gradient(mean))
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution_mean": mean,
            "latent_distribution_variance": tf.maximum(scale, 0.1),
        }
        return q_latent, q_hyper_latent, state4bpp

    @tf.function
    def decode(self, input):
        output = []
        for resnet, up in self.dec:
            input = resnet(input)
            input = up(input)
            output.append(input)
        return output[::-1]

    @tf.function
    def bpp(self, shape, state4bpp, training):
        B, _, H, W = self.img_batch_shape # shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution_mean = state4bpp["latent_distribution_mean"]
        latent_distribution_variance = state4bpp["latent_distribution_variance"]
        if training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", tf.stop_gradient(latent_distribution_mean))
        hyper_rate = -tf.math.log(self.prior.likelihood(q_hyper_latent)) / tf.math.log(2.0)
        # cond_rate = -tf.math.log(latent_distribution.likelihood(q_latent)) / tf.math.log(2.0)
        cond_rate =  -tf.math.log(self.normal_distribution_likelihood(q_latent, latent_distribution_mean, latent_distribution_variance)) / tf.math.log(2.0)
        bpp = (tf.reduce_sum(hyper_rate, axis=[1, 2, 3]) + tf.reduce_sum(cond_rate, axis=[1, 2, 3])) / tf.cast(H*W, cond_rate.dtype)
        return bpp

    @tf.function
    def normal_distribution_likelihood(self, x, mean, variance, min=1e-9):
        x = tf.abs(x - mean)
        upper = self.std_cdf((0.5 - x) / variance)
        lower = self.std_cdf((-0.5 - x) / variance)
        return lower_bound(upper - lower, min)

    @tf.function
    def std_cdf(self, inputs):
        half = 0.5
        const = -(2 ** -0.5)
        return half * tf.math.erfc(const * inputs)

    @tf.function
    def call(self, input, training):
        q_latent, q_hyper_latent, state4bpp = self.encode(input)
        bpp = self.bpp(self.img_batch_shape, state4bpp, training=training)
        output = self.decode(q_latent) #dummy  # output = self.decode(state4bpp["latent"])
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }


class ResnetCompressor(Compressor):
    def __init__(
        self,
            batch_size,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
    ):
        super(ResnetCompressor, self).__init__(
            batch_size,
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels
        )

        self.batch_size = batch_size

        self.x_start_shape = (self.batch_size, 2, 32, 32)
        self.noise_shape = (self.batch_size, 2, 32, 32)
        self.img_batch_shape = (self.batch_size, 2, 32, 32)

        self.build_network()

    def build_network(self):
        self.enc = []
        self.dec = []
        self.hyper_enc = []
        self.hyper_dec = []

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append([
                ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                Downsample(dim_out)
            ])

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append([
                ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                Upsample(dim_out if not is_last else dim_in, dim_out)
            ])

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            if ind == 0:
                self.hyper_enc.append([
                    ZeroPadding2D(padding=1, data_format="channels_first"),
                    Conv2D(dim_out, 3, 1, 'valid', data_format="channels_first"),
                    LeakyReLU(0.2) if not is_last else tf.keras.layers.Identity()
                ])
            else:
                self.hyper_enc.append([
                    ZeroPadding2D(padding=2, data_format="channels_first"),
                    Conv2D(dim_out, 5, 2, 'valid',data_format="channels_first"),
                    LeakyReLU(0.2) if not is_last else tf.keras.layers.Identity()
                ])

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            if ind == 0:
                self.hyper_dec.append([
                    ZeroPadding2D(padding=1, data_format="channels_first"),
                    Conv2D(dim_out, 3, 1, 'valid', data_format="channels_first"),
                    LeakyReLU(0.2) if not is_last else tf.keras.layers.Identity()
                ])
            else:
                self.hyper_dec.append([
                    ZeroPadding2D(padding=0, data_format="channels_first"),
                    Conv2DTranspose(dim_out, 5, 2, 'same', output_padding=1, data_format="channels_first"),
                    LeakyReLU(0.2) if not is_last else tf.keras.layers.Identity()
                ])





if __name__ == "__main__":
    print("test")
    random_image = tf.ones((100, 2, 32,32))
    resnet_compressor = ResnetCompressor()