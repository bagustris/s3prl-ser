import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .convert import load_converted_model


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        model, task_cfg = load_converted_model(ckpt)
        self.model = model
        self.wav_normalize = task_cfg.normalize

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

        self._init_layerdrop = self.model.encoder.layerdrop

    @property
    def layer_drop(self):
        return self.model.encoder.layerdrop

    def set_layer_drop(self, layerdrop: float = None):
        if isinstance(layerdrop, float):
            self.model.encoder.layerdrop = layerdrop
        elif layerdrop is None:
            self.model.encoder.layerdrop = self._init_layerdrop
        else:
            raise ValueError("layerdrop can only be float or None")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(padded_wav, wav_padding_mask)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
