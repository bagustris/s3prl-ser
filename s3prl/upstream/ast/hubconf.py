from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def ast(
    refresh: bool = False,
    window_secs: float = 10.24,
    stride_secs: float = 10.24,
    **kwds,
):
    kwds["ckpt"] = _urls_to_filepaths(
        "https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1",
        refresh=refresh,
    )
    return _UpstreamExpert(window_secs=window_secs, stride_secs=stride_secs, **kwds)
