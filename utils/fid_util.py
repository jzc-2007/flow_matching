from absl import logging
import time
import functools

import numpy as np

import jax
import jax.numpy as jnp
from .jax_fid import inception
from .jax_fid import resize


import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image
import torch.nn.functional as F

from .jax_fid.fid import compute_frechet_distance
compute_fid = compute_frechet_distance


def build_jax_inception(batch_size=200):
    # jax model
    logging.info("Initializing InceptionV3")
    model = inception.InceptionV3(pretrained=True)
    inception_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 16, 16, 3)))
    logging.info("Initialized InceptionV3")
    inception_fn = jax.jit(functools.partial(model.apply, train=False))

    fake_x = np.zeros((batch_size, 299, 299, 3), dtype=np.float32)
    lowered = inception_fn.lower(inception_params, jax.lax.stop_gradient(fake_x))
    logging.info('Start compiling inception_fn...')
    t_start = time.time()
    compiled = lowered.compile()
    logging.info(f'End compiling: {(time.time() - t_start):.4f} seconds.')
    inception_fn = compiled

    inception_net = {"params": inception_params, "fn": inception_fn}
    return inception_net


def get_reference(cache_path, inception_net, batch_size=200, num_samples=50000):
    # Save ref_mu and ref_sigma to npz file
    if not os.path.exists(cache_path):
        logging.info("Computing ref_mu and ref_sigma...")
        transforms = torchvision.transforms.PILToTensor()
        train_ds = CIFAR10('~/cache', train=True, download=True, transform=transforms)
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=12,
        )
        all_real = []
        for x in train_dataloader:
            all_real.append(x[0].numpy().transpose(0,2,3,1))
        all_real = np.concatenate(all_real)

        ref_mu, ref_sigma = compute_jax_fid(
            all_real[: num_samples],
            inception_net,
            batch_size=batch_size
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, ref_mu=ref_mu, ref_sigma=ref_sigma)
        logging.info(f"Saved ref_mu and ref_sigma to {cache_path}")
        os.system('md5sum ' + cache_path)
    else:
        logging.info(f"Loading ref_mu and ref_sigma from {cache_path}")
        os.system('md5sum ' + cache_path)
        # e33f43d9e68c76396d322d4a8942f904: cifar10_jax_stats.npz
        # 0a87a113394cae12e0f1f75d9070d842: cifar10_jax_stats_20240820.npz
        with np.load(cache_path) as data:
            if "ref_mu" in data:
                ref_mu, ref_sigma = data["ref_mu"], data["ref_sigma"]
            elif "mu" in data:
                ref_mu, ref_sigma = data["mu"], data["sigma"]
            else:
                raise NotImplementedError

    ref = {"mu": ref_mu, "sigma": ref_sigma}
    return ref


def compute_jax_fid(
    samples,
    inception_net,
    batch_size=200,
    num_workers=12,
    mode = "legacy_tensorflow"
):
    inception_fn = inception_net["fn"]
    inception_params = inception_net["params"]

    num_samples = len(samples)

    pad = int(np.ceil(num_samples / batch_size)) * batch_size - num_samples
    samples = np.concatenate([samples, np.zeros((pad, *samples.shape[1:]), dtype=np.uint8)])
    assert len(samples) % batch_size == 0

    dataset = ResizeDataset(samples, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    l_feats = []
    for i, x in enumerate(dataloader):
        if i % 50 == 0:
            logging.info(f"Evaluating {i} / {len(dataloader)}: {list(x.shape)}")
        x = resize.forward(x)  # Kaiming: match the Pytorch version
        x = x.numpy().transpose(0,2,3,1)
        pred = inception_fn(inception_params, jax.lax.stop_gradient(x))
        pred = pred.squeeze(axis=1).squeeze(axis=1)
        l_feats.append(pred)
    np_feats = np.concatenate(l_feats)
    np_feats = np_feats[:num_samples]
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)

    return mu, sigma


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


# this is from the cleanfid package
def build_resizer(mode):
    if mode == "clean":
        return make_resizer("PIL", False, "bicubic", (299,299))
    # if using legacy tensorflow, do not manually resize outside the network
    elif mode == "legacy_tensorflow":
        return lambda x: x
    elif mode == "legacy_pytorch":
        return make_resizer("PyTorch", False, "bilinear", (299, 299))
    else:
        raise ValueError(f"Invalid mode {mode} specified")


"""
Construct a function that resizes a numpy image based on the
flags passed in.
"""
def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX
        }
        def func(x):
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=name_to_filter[filter])
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
            return x
    elif library == "PIL" and not quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX
        }
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=name_to_filter[filter])
            return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    elif library == "TensorFlow":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        import tensorflow as tf
        def func(x):
            x = tf.constant(x)[tf.newaxis, ...]
            x = tf.image.resize(x, output_size, method=filter)
            x = x[0, ...].numpy().clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    elif library == "OpenCV":
        raise NotImplementedError
        # import cv2
        # name_to_filter = {
        #     "bilinear": cv2.INTER_LINEAR,
        #     "bicubic": cv2.INTER_CUBIC,
        #     "lanczos": cv2.INTER_LANCZOS4,
        #     "nearest": cv2.INTER_NEAREST,
        #     "area": cv2.INTER_AREA
        # }
        # def func(x):
        #     x = cv2.resize(x, output_size, interpolation=name_to_filter[filter])
        #     x = x.clip(0, 255)
        #     if quantize_after:
        #         x = x.astype(np.uint8)
        #     return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func