from nnAudio import features
from .SimpleLightningBase import SimpleLightningBase

class STFTCNN(SimpleLightningBase):
    def __init__(self, stft_n_fft, stft_hop_length, stft_sr) -> None:
        
        super().__init__()

        self.stft = features.STFT(
            n_fft         = stft_n_fft,
            hop_length    = stft_hop_length,
            sr            = stft_sr,
            output_format ="Magnitude"
        )