import tensorflow as tf
from tensorflow.keras import layers, Model
from .utils import exists, default
from .network_components import (
    LayerNorm,
    Residual,
    # SinusoidalPosEmb,
    Upsample,
    Downsample,
    PreNorm,
    LinearAttention,
    # Block,
    ResnetBlock,
    ImprovedSinusoidalPosEmb
)

class Unet(Model):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        context_dim_mults=(1, 2, 3, 3),
        channels=3,
        context_channels=3,
        with_time_emb=True,
        embd_type="01"
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        context_dims = [context_channels, *map(lambda m: dim * m, context_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.embd_type = embd_type

        if with_time_emb:
            if embd_type == "01":
                time_dim = dim
                self.time_mlp = tf.keras.Sequential([
                    layers.Dense(dim * 4, activation=tf.keras.activations.gelu),
                    layers.Dense(dim)
                ])
            elif embd_type == "index":
                time_dim = dim
                self.time_mlp = tf.keras.Sequential([
                    ImprovedSinusoidalPosEmb(time_dim // 2),
                    layers.Dense(time_dim // 2 + 1, activation=tf.keras.activations.gelu),
                    layers.Dense(time_dim * 4, activation=tf.keras.activations.gelu),
                    layers.Dense(time_dim)
                ])
            else:
                raise NotImplementedError
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                ResnetBlock(
                    dim_in + context_dims[ind]
                    if (not is_last) and (ind < (len(context_dims) - 1))
                    else dim_in,
                    dim_out,
                    time_dim,
                    True if ind == 0 else False
                ),
                ResnetBlock(dim_out, dim_out, time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else tf.keras.layers.Identity(),
            ])

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                ResnetBlock(dim_out * 2, dim_in, time_dim),
                ResnetBlock(dim_in, dim_in, time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else tf.keras.layers.Lambda(lambda x: x),
            ])

        out_dim = default(out_dim, channels)
        self.final_conv = tf.keras.Sequential([
            #LayerNorm(dim), #
            tf.keras.layers.LayerNormalization(axis=[1,2,3], epsilon=1e-5),
            layers.ZeroPadding2D(3, data_format="channels_first"),
            layers.Conv2D(out_dim, 7, padding='valid', data_format="channels_first")
        ])

        # self.randomconv1 = layers.Conv2D(out_dim, 7, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv2 = layers.Conv2D(out_dim, 7, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv3 = layers.Conv2D(out_dim, 7, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv4 = layers.Conv2D(out_dim, 2, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv5 = layers.Conv2D(out_dim, 2, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv6 = layers.Conv2D(out_dim, 2, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv7 = layers.Conv2D(out_dim, 2, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv8 = layers.Conv2D(out_dim, 2, padding='same', data_format="channels_first", activation="relu")
        # self.randomconv9 = layers.Conv2D(out_dim, 2, padding='same', data_format="channels_first")
        # self.randomdense = layers.Dense(2*32*32)
        # self.randomdense2 = layers.Dense(2*32*32)

    def encode(self, x, t, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
        #for idx, (backbone, backbone2,  downsample) in enumerate(self.downs):
            x = tf.concat([x, context[idx]], axis=1) if idx < len(context) else x
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        return x, h

    def decode(self, x, h, t):
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for backbone, backbone2, attn, upsample in self.ups:
        #for backbone, backbone2,  upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=1)
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)

    @tf.function
    def call(self, x, time=None, context=None):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        #original_x = x
        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)


        # y = y * 0.0 + original_x
        # context_reshape = layers.Flatten()(context[0])
        # context_reshape = self.randomdense(context_reshape)
        # time_reshape = layers.Flatten()(t)
        # time_reshape = self.randomdense2(time_reshape)
        # context_reshape = tf.reshape(context_reshape, (100,2,32,32))
        # time_reshape = tf.reshape(time_reshape, (100,2,32,32))
        # y = self.randomconv1(y+context_reshape+time_reshape)
        # y = self.randomconv2(y)
        # y = self.randomconv3(y+context_reshape)
        # y = self.randomconv4(y)
        # y = self.randomconv5(y)
        # y = self.randomconv6(y+context_reshape)
        # y = self.randomconv7(y)
        # y = self.randomconv8(y)
        # y = self.randomconv9(y)
        # return y

