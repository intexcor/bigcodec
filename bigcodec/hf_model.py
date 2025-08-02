import torch
from huggingface_hub import PyTorchModelHubMixin
from .modules import CodecEncoder, CodecDecoder


class BigCodec(torch.nn.Module, PyTorchModelHubMixin, license="mit"):

    def __init__(self, ngf, use_rnn, rnn_bidirectional, rnn_num_layers, up_ratios, dilations, out_channels,
                 codebook_size, vq_num_quantizers, codebook_dim):
        super().__init__()
        self.encoder = CodecEncoder(
            ngf=ngf,
            use_rnn=use_rnn,
            rnn_bidirectional=rnn_bidirectional,
            rnn_num_layers=rnn_num_layers,
            up_ratios=up_ratios,
            dilations=dilations,
            out_channels=out_channels
        )
        self.decoder = CodecDecoder(
            in_channels=out_channels,
            vq_num_quantizers=vq_num_quantizers,
            vq_dim=out_channels,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            up_ratios=tuple(reversed(up_ratios)),
            dilations=dilations
        )

    def forward(self, x):
        latents = self.encoder(x)
        audio, *_ = self.decoder(latents, vq=False)
        return audio

    def encode(self, x):
        vq_emb = self.encoder(x)
        _, vq, _ = self.decoder(vq_emb, vq=True)
        return vq

    def decode(self, vq):
        vq_emb = self.decoder.vq2emb(vq.transpose(1, 2)).transpose(1, 2)
        audio = self.decoder(vq_emb, vq=False)
        return audio
