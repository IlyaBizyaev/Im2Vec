from .base import *
from .vanilla_vae import *
from .vector_vae import VectorVAE
from .vector_vae_nlayers import VectorVAEnLayers


VAE_MODELS = {
    'VanillaVAE': VanillaVAE,
    'VectorVAE': VectorVAE,
    'VectorVAEnLayers': VectorVAEnLayers
}
