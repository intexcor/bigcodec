## BigCodec

Python library for bigcodec. This is something that the author of bigcodec didn't do, so I'm doing it for him.

# Setup

pip install bigcodec

# Usage

import torch.accelerator
from bigcodec import BigCodec
import torchaudio

codec = BigCodec.from_pretrained("intexcp/bigcodec")

wav = torchaudio.load("enc.wav")[0]
wav = wav.unsqueeze(0)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
codec = codec.to(device)
wav = wav.to(device)
with torch.no_grad():
    enc = codec.encode(wav)
    dec = codec.decode(enc)
print(enc)
torchaudio.save("dec.wav", dec.squeeze(0).cpu(), 16000, encoding="PCM_F")

