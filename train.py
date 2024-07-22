
import os
import argparse


parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--gpu", type=int, default=2, help="cuda device id")
parser.add_argument("--random_seed", type=int, default=42, help="random seed")
parser.add_argument("--beta", type=float, default=0.003, help="beta for bitrate control")
parser.add_argument("--z_channels", type=int, default=2)
parser.add_argument("--ae_dim_mult", type=int, nargs='+', default=[1,2])
parser.add_argument("--ae_base_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay", type=float, default=0.8)
parser.add_argument("--minf", type=float, default=0.2)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--n_step", type=int, default=100000)
parser.add_argument("--scheduler_checkpoint_step", type=int, default=100)
parser.add_argument("--log_checkpoint_step", type=int, default=50)
parser.add_argument("--load_model", default=False)
parser.add_argument("--load_step", default=False)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument('--pred_mode', type=str, default='noise', help='prediction mode')
parser.add_argument('--loss_type', type=str, default='l2', help='type of loss')
parser.add_argument('--iteration_step', type=int, default=4, help='number of iterations')
parser.add_argument('--sample_steps', type=int, default=4, help='number of steps for sampling (for validation)')
parser.add_argument('--embed_dim', type=int, default=64, help='dimension of embedding')
parser.add_argument('--embd_type', type=str, default="01", help='timestep embedding type')
parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 3, 4], help='dimension multipliers')
parser.add_argument('--hyper_dim_mults', type=int, nargs='+', default=[4, 4, 4], help='hyper dimension multipliers')
parser.add_argument('--context_dim_mults', type=int, nargs='+', default=[1, 2], help='context dimension multipliers')
parser.add_argument('--reverse_context_dim_mults', type=int, nargs='+', default=[2, 1], help='reverse context dimension multipliers')
parser.add_argument('--context_channels', type=int, default=8, help='number of context channels')
parser.add_argument('--use_weighted_loss', action='store_true', help='if use weighted loss')
parser.add_argument('--weight_clip', type=int, default=5, help='snr clip for weighted loss')
parser.add_argument('--use_mixed_precision', action='store_true', help='if use mixed precision')
parser.add_argument('--clip_noise', action='store_true', help='if clip the noise during sampling')
parser.add_argument('--val_num_of_batch', type=int, default=1, help='number of batches for validation')
parser.add_argument('--additional_note', type=str, default='', help='additional note')
parser.add_argument('--var_schedule', type=str, default='cosine', help='variance schedule')
parser.add_argument('--aux_loss_type', type=str, default='l2', help='type of auxiliary loss')
parser.add_argument("--aux_weight", type=float, default=0, help="weight for aux loss")

parser.add_argument("--data_name", type=str, default="setup_your_own_dataset", help="name of dataset", choices=["setup_your_own_dataset"])
parser.add_argument("--data_root", type=str, default="", help="root of dataset")
parser.add_argument("--params_root", type=str, default="")
parser.add_argument("--tensorboard_root", type=str, default="")
parser.add_argument("--ae_path", type=str, default="", help="path to ae model")

parser.add_argument("--use_aux_loss_weight_schedule", action="store_true", help="if use aux loss weight schedule")

config = parser.parse_args()



gpu_num = int(config.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
random_seed = config.random_seed


import random
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
from tensorflow import keras
import numpy as np

tf.config.run_functions_eagerly(False)
#tf.compat.v1.disable_eager_execution()


# os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from data import load_data
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from modules.compress_modules import ResnetCompressor


data_config = {
    "dataset_name": config.data_name,
    "data_path": config.data_root,
    "sequence_length": 1,
    "img_size": 32,
    "img_channel": 2,
    "add_noise": False,
    "img_hz_flip": False,
}

model_name = (
    f"{'latent' if len(config.ae_path)>0 else 'image'}-{config.loss_type}-{'use_weight'+str(config.weight_clip) if config.use_weighted_loss else 'no_weight'}-{data_config['dataset_name']}"
    f"-d{config.embed_dim}-t{config.iteration_step}-b{config.beta}"
    f"-{config.pred_mode}-{config.var_schedule}-{config.embd_type}-{'mixed' if config.use_mixed_precision else 'float32'}-{'auxschedule-' if config.use_aux_loss_weight_schedule else ''}aux{config.aux_weight}{config.aux_loss_type if config.aux_weight>0 else ''}{config.additional_note}"
)

print('model name:')
print(model_name)


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def main():

    train_data, val_data = load_data(
        data_config,
        config.batch_size,
        num_workers=config.n_workers,
    )
    data_generator = None

    context_model = ResnetCompressor(
        batch_size=config.batch_size,
        dim=config.embed_dim,
        dim_mults=config.context_dim_mults,
        reverse_dim_mults=config.reverse_context_dim_mults,
        hyper_dims_mults=config.hyper_dim_mults,
        channels=data_config["img_channel"],
        out_channels=config.context_channels,
    )

    ae_fn = None
    
    denoise_model = Unet(
        dim=config.embed_dim,
        channels=data_config["img_channel"] if ae_fn is None else config.z_channels,
        context_channels=config.context_channels,
        dim_mults=config.dim_mults,
        context_dim_mults=reversed(config.reverse_context_dim_mults),
        embd_type=config.embd_type,
    )

    input_img_shape = (2, 32, 32)

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        input_img_shape=input_img_shape,
        batch_size=config.batch_size,
        ae_fn=ae_fn,
        num_timesteps=config.iteration_step,
        loss_type=config.loss_type,
        lagrangian=config.beta,
        pred_mode=config.pred_mode,
        aux_loss_weight=config.aux_weight,
        aux_loss_type=config.aux_loss_type,
        var_schedule=config.var_schedule,
        use_loss_weight=config.use_weighted_loss,
        loss_weight_min=config.weight_clip,
        use_aux_loss_weight_schedule=config.use_aux_loss_weight_schedule
    )

    # loss, aloss = diffusion(random_input)


    trainer = Trainer(
        rank=config.gpu,
        sample_steps=config.sample_steps,
        diffusion_model=diffusion,
        train_dl=train_data,
        val_dl=val_data,
        data_generator=data_generator,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.params_root, f"{model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{model_name}/"),
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
    )

    if config.load_model:
        print('loaded')
        trainer.load(idx=0, load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":
    main()
