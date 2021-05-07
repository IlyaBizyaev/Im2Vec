from .base import *
from .vanilla_vae import *
from .vector_vae import VectorVAE, raster_verbose, HIGH, LOW
from .vector_vae_nlayers import VectorVAEnLayers


VAE_MODELS = {
    'VanillaVAE': VanillaVAE,
    'VectorVAE': VectorVAE,
    'VectorVAEnLayers': VectorVAEnLayers
}


def make_model(config):
    return VAE_MODELS[config['model_params']['name']](
        img_size=config['exp_params']['img_size'],
        **config['model_params']
    )