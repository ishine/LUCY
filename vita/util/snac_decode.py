import torch
import numpy as np
import soundfile as sf
from snac import SNAC
snac_sr = 24_000
cache_dir = "/path/to/cache/dir"
with open("/path/to/snac/file", "r") as f:
    for l in f:
        codes = l.strip()
        break
codes = np.array(list(map(int, codes.strip().split())))
codes = codes.reshape(-1, 7)
device = "cuda:0"
codes_12hz, codes_24hz, codes_48hz = codes[:, 0:1], codes[:, 1:3], codes[:, 3:7]
full_codes = [
    torch.LongTensor(codes_12hz.reshape(1,-1)).to(device), 
    torch.LongTensor(codes_24hz.reshape(1,-1)).to(device), 
    torch.LongTensor(codes_48hz.reshape(1,-1)).to(device)
]
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=cache_dir).eval().to(device)
with torch.inference_mode():
    audio_hat = model.decode(full_codes).view(-1)
wav = audio_hat.view(-1).cpu().numpy()
sf.write('output.wav', wav, snac_sr)
