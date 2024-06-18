import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

import importlib
# importlib.import_module('conformer')
# from conformer.decoder import DecoderRNNT
# from conformer.encoder import ConformerEncoder
# from conformer.modules import Linear


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs
    ):
        super(CNNSelfAttention, self).__init__()
        self.model_seq = nn.Sequential(
            nn.AvgPool1d(kernel_size, pooling, padding),
            nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
        )
        self.pooling = SelfAttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs,
    ):
        super(FCN, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 96, 11, stride=4, padding=5),
            nn.LocalResponseNorm(96),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(96, 256, 5, padding=2),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(384, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, 3, padding=1),
            nn.LocalResponseNorm(256),
            nn.MaxPool1d(3, 2),
        )
        self.pooling = SelfAttentionPooling(256)
        self.out_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class DeepNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs,
    ):
        super(DeepNet, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 10, 9),
            nn.ReLU(),
            nn.Conv1d(10, 10, 5),
            nn.ReLU(),
            nn.Conv1d(10, 10, 3),
            nn.MaxPool1d(3, 1),
            nn.BatchNorm1d(10, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(10, 40, 3),
            nn.ReLU(),
            nn.Conv1d(40, 40, 3),
            nn.MaxPool1d(2, 1),
            nn.BatchNorm1d(40, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(40, 80, 10),
            nn.ReLU(),
            nn.Conv1d(80, 80, 1),
            nn.MaxPool1d(2, 1),
            nn.BatchNorm1d(80, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(80, 80, 1),
        )
        self.pooling = SelfAttentionPooling(80)
        self.out_layer = nn.Sequential(
            nn.Linear(80, 30),
            nn.ReLU(),
            nn.Linear(30, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        decoder_dim (int, optional): Dimension of conformer decoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_decoder_layers (int, optional): Number of decoder layers
        decoder_rnn_type (str, optional): type of RNN cell
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        decoder_dropout_p (float, optional): Probability of conformer decoder dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
            self,
            num_classes: int = 4,
            input_dim: int = 80,
            encoder_dim: int = 512,
            decoder_dim: int = 640,
            num_encoder_layers: int = 17,
            num_decoder_layers: int = 1,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            decoder_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            decoder_rnn_type: str = "lstm",
    ) -> None:
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.decoder = DecoderRNNT(
            num_classes=num_classes,
            hidden_state_dim=decoder_dim,
            output_dim=encoder_dim,
            num_layers=num_decoder_layers,
            rnn_type=decoder_rnn_type,
            dropout_p=decoder_dropout_p,
        )
        self.fc = Linear(encoder_dim << 1, num_classes, bias=False)

    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)

        return outputs

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        """
        Decode `encoder_outputs`.

        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor(
            [[self.decoder.sos_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_states=hidden_state)
            step_output = self.joint(
                encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor(
                [[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs


class DeepModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_type,
        pooling,
        **kwargs
    ):
        super(DeepModel, self).__init__()
        self.pooling = pooling
        self.model = eval(model_type)(
            input_dim=input_dim, output_class_num=output_dim, pooling=pooling, **kwargs)

    def forward(self, features, features_len):
        attention_mask = [
            torch.ones(math.ceil((l / self.pooling)))
            for l in features_len
        ]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        attention_mask = (1.0 - attention_mask) * -100000.0
        attention_mask = attention_mask.to(features.device)
        predicted = self.model(features, attention_mask)
        return predicted, None
