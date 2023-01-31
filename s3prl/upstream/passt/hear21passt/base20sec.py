import torch

from .models.passt import get_model as get_model_passt
from .models.preprocess import AugmentMelSTFT
from .wrapper import PasstBasicWrapper


def load_model(model_path="", mode="all", scene_hop=5000, **kwds):
    """
    scene_hop: The hop size for the ovelaping windows
    in case the scene audio lenght is larger than 20 seconds.
    Returns:
    model: wrapped PaSST model that can take up to 20 seconds
     of audio without averaging the embeddings.
    """
    model = get_basic_model(mode=mode, scene_hop=scene_hop, **kwds)
    return model


def get_scene_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
    timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """
    return get_basic_timestamp_embeddings(audio, model)


def get_basic_timestamp_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
    timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """
    return model.get_timestamp_embeddings(audio)


def get_basic_model(**kwargs):
    mel = AugmentMelSTFT(
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        freqm=48,
        timem=192,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=10,
        fmax_aug_range=2000,
    )

    net = get_model_passt(arch="passt_20sec", input_tdim=2000)
    model = PasstBasicWrapper(mel=mel, net=net, max_model_window=20000, **kwargs)
    return model
