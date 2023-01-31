# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2/hubconf.py ]
#   Synopsis     [ the wav2vec 2.0 torch hubconf ]
#   Author       [ S3PRL / Kushal Lakhotia ]
"""*********************************************************************************************"""

import logging
import os
import time
from pathlib import Path

from filelock import FileLock

from s3prl.util.download import _urls_to_filepaths

from .convert import load_and_convert_fairseq_ckpt
from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0


def wav2vec2_custom(
    ckpt: str,
    legacy: bool = False,
    fairseq: bool = False,
    refresh: bool = False,
    **kwargs,
):
    assert not (legacy and fairseq), (
        "The option 'legacy' will directly load a fairseq checkpoint, "
        "while the option 'fairseq' will first convert the fairseq checkpoint to "
        "be fairseq indenpendent and then load the checkpoint. "
        "These two options cannot be used jointly."
    )

    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    if fairseq:
        ckpt: Path = Path(ckpt)
        converted_ckpt = ckpt.parent / f"{ckpt.stem}.converted.pt"
        lock_file = Path(str(converted_ckpt) + ".lock")

        logger.info(f"Converting a fairseq checkpoint: {ckpt}")
        logger.info(f"To: {converted_ckpt}")

        with FileLock(str(lock_file)):
            if not converted_ckpt.is_file() or (
                refresh and (time.time() - os.path.getmtime(ckpt)) > NEW_ENOUGH_SECS
            ):
                load_and_convert_fairseq_ckpt(ckpt, converted_ckpt)

        ckpt = converted_ckpt

    assert os.path.isfile(ckpt)
    if legacy:
        return _LegacyUpstreamExpert(ckpt, **kwargs)
    else:
        return _UpstreamExpert(ckpt, **kwargs)


def wav2vec2_local(*args, **kwargs):
    return wav2vec2_custom(*args, **kwargs)


def wav2vec2_url(*args, **kwargs):
    return wav2vec2_custom(*args, **kwargs)


def wav2vec2(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_base_960(refresh=refresh, *args, **kwargs)


def wav2vec2_base_960(refresh=False, legacy=False, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_small.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_large_960(refresh=False, legacy=False, **kwargs):
    """
    The Large model trained on LibriSpeech 960 hours of data
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/libri960_big.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_large_ll60k(refresh=False, legacy=False, **kwargs):
    """
    The Large model trained on Libri-light 60k hours of data
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_vox_new.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_large_lv60_cv_swbd_fsh(refresh=False, legacy=False, **kwargs):
    """
    The Large model trained on Libri-Light 60k hours + CommonVoice + Switchboard + Fisher
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/w2v_large_lv_fsh_swbd_cv.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def xlsr_53(refresh=False, legacy=False, **kwargs):
    """
    The wav2vec 2.0 model trained on multilingual presented in https://arxiv.org/abs/2006.13979
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr_53_56k.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def xls_r_300m(refresh=False, legacy=False, **kwargs):
    """
    XLS-R, this smallest size has the same parameters as the Largs model of wav2vec 2.0 and HuBERT
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr2_300m.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def xls_r_1b(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr2_960m_1000k.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def xls_r_2b(refresh=False, legacy=False, **kwargs):
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr2_2B_1000k.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_conformer_relpos(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/conformer/wav2vec2/librilight/LL_relpos_PT_no_FT"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/LL_relpos_PT_no_FT.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_conformer_rope(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/conformer/wav2vec2/librilight/LL_rope_PT_no_FT"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/LL_rope_PT_no_FT.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_large_voxpopuli_100k(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_100k.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_large_100k.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_base_s2st_es_voxpopuli(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/transformer_B.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_base_s2st_es_voxpopuli.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


# FIXME: the official fairseq checkpoint link is down currently
# def wav2vec2_large_s2st_es_voxpopuli(refresh=False, legacy=False, **kwargs):
#     kwargs[
#         "ckpt"
#     ] = "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/transformer_L.pt"
#     if not legacy:
#         kwargs[
#             "ckpt"
#         ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2-large-s2st-es-voxpopuli.pt"
#     return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_conformer_large_s2st_es_voxpopuli(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/conformer_L.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_conformer_large_s2st_es_voxpopuli.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_base_s2st_en_librilight(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/en/transformer_B.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_base_s2st_en_librilight.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)


def wav2vec2_conformer_large_s2st_en_librilight(refresh=False, legacy=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/en/conformer_L.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_conformer_large_s2st_en_librilight.pt"
    return wav2vec2_custom(refresh=refresh, legacy=legacy, **kwargs)
