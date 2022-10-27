from matplotlib import pyplot as plt


import numpy as np
import random
from scipy.io.wavfile import write
import librosa
import torch

from mellotron.hparams import create_hparams, AttrDict
from mellotron.model import Tacotron2 as Mellotron
from mellotron.layers import TacotronSTFT as MellotronSTFT
from waveglow.glow import WaveGlow
from waveglow.denoiser import Denoiser
from mellotron.data_utils import TextMelCollate
from mellotron.text import cmudict, text_to_sequence
from mellotron.yin import compute_yin

from typing import Optional, Tuple


def load_tts(
        tts_model_checkpoint_path: str,
        device: Optional[torch.device] = None,
) -> Tuple[Mellotron, MellotronSTFT, AttrDict]:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load hyper-parameters
    hparams: AttrDict = create_hparams()
    # Create STFT PyTorch layer
    stft = MellotronSTFT(
        hparams.filter_length,
        hparams.hop_length,
        hparams.win_length,
        hparams.n_mel_channels,
        hparams.sampling_rate,
        hparams.mel_fmin,
        hparams.mel_fmax
    )
    # Create Mellotron TTS instance and load weights
    mellotron: Mellotron = Mellotron(hparams).to(device).eval()
    mellotron.load_state_dict(torch.load(tts_model_checkpoint_path, map_location=device)['state_dict'])

    return mellotron, stft, hparams


def load_vocoder(
        vocoder_model_checkpoint_path: str,
        device: Optional[torch.device] = None
) -> Tuple[WaveGlow, Denoiser]:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load vocoder
    waveglow: WaveGlow = torch.load(vocoder_model_checkpoint_path, map_location=device)['model'].eval()
    # Load denoiser
    denoiser: Denoiser = Denoiser(waveglow).to(device).eval()

    return waveglow, denoiser


def load_arpabet_dict(dict_path: str):
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    arpabet_dict = cmudict.CMUDict(dict_path)

    return arpabet_dict


def _get_mel_spec(
        stft: MellotronSTFT,
        hparams: AttrDict,
        audio_path: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None
) -> torch.FloatTensor:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/data_utils.py
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enforce either audio path or audio data but not both
    assert (audio_path is not None and audio is None) or (audio_path is None and audio is not None)

    # Load audio file
    if audio_path is not None:
        audio, _ = librosa.core.load(audio_path, sr=hparams.sampling_rate)
    # Convert to PyTorch format
    audio = torch.from_numpy(audio)
    # Normalize (If not already in [-1; 1] range)
    audio_norm = audio / hparams.max_wav_value if audio.max() > 1.0 or audio.min() < -1.0 else audio
    audio_norm = audio_norm.unsqueeze(0)
    # Compute Mel spectrogram using utility layer
    mel_spec = stft.mel_spectrogram(audio_norm)
    mel_spec = mel_spec.to(device)

    return mel_spec


def _get_f0(
        hparams: AttrDict,
        audio_path: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        stft: Optional[MellotronSTFT] = None,
        mel_spec: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None
) -> torch.FloatTensor:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/data_utils.py
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enforce either audio path or audio, but not both, and either STFT or Mel spectrogram, but not both
    assert (audio_path is not None and audio is None) or (audio_path is None and audio is not None)
    assert (stft is not None and mel_spec is None) or (stft is None and mel_spec is not None)

    # Load audio file
    if audio_path is not None:
        audio, _ = librosa.core.load(audio_path, sr=hparams.sampling_rate)

    # Get input Mel Spectrogram
    mel_spec = mel_spec if mel_spec is not None else _get_mel_spec(stft, hparams, audio, device=device).cpu().numpy()

    # Compute F0
    f0, harmonic_rates, argmins, times = compute_yin(
        audio,
        hparams.sampling_rate,
        hparams.filter_length,
        hparams.hop_length,
        hparams.f0_min,
        hparams.f0_max,
        hparams.harm_thresh
    )
    # Apply zero padding on the sides
    pad = int((hparams.filter_length / hparams.hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad
    # Convert to PyTorch  format
    f0 = np.array(f0, dtype=np.float32)
    f0 = torch.from_numpy(f0)[None]
    f0 = f0[:, :mel_spec.shape[-1]]

    return f0


@torch.no_grad()
def get_prosody_embedding(
        reference_audio_path: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: AttrDict,
        device: Optional[torch.device] = None
) ->torch.FloatTensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Mel spectrogram
    mel_spec = _get_mel_spec(reference_audio_path, stft, hparams, device=device)
    # Compute prosody embedding via reference encoder
    prosody_embedding = mellotron.gst.encoder(mel_spec)

    return prosody_embedding


@torch.no_grad()
def get_gst(
        reference_audio_path: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: AttrDict,
        device: Optional[torch.device] = None
) -> torch.FloatTensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Mel spectrogram
    mel_spec = _get_mel_spec(reference_audio_path, stft, hparams, device=device)

    ...


def encode_input(
        text: str,
        reference_audio_path: str,
        arpabet_dict,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: AttrDict,
        speaker_id: Optional[int] = None,
        device: Optional[torch.device] = None
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create
    data_collate = TextMelCollate(1)

    # Encode text
    text = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict), device=device)[None, :]

    # Encode audio
    audio, _ = librosa.core.load(reference_audio_path, sr=hparams.sampling_rate)
    # Compute Reference audio Mel spectrogram
    mel_spec = _get_mel_spec(stft, hparams, audio=audio, device=device)
    # Compute pitch contour
    f0 = _get_f0(hparams, audio=audio, mel_spec=mel_spec.squeeze().cpu().numpy(), device=device)
    # Encode speaker ID
    # If speaker ID is none sample one randomly from model
    if speaker_id is None:
        speaker_id = random.randint(0, mellotron.speaker_embedding.num_embeddings)
    speaker_id = torch.LongTensor([speaker_id], device=device)

    (text, style_input, speaker_ids, f0s), _ = mellotron.parse_batch(data_collate([(text, mel_spec, speaker_id, f0)]))

    return text, style_input, speaker_ids, f0s


def _plot_mel_f0_alignment(
        mel_source, mel_outputs_postnet, f0s, alignments, figsize=(16, 16), output_path: Optional = None
):
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto', origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)


@torch.no_grad()
def synthesise_speech(
        text: str,
        reference_audio_path: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: AttrDict,
        waveglow: WaveGlow,
        denoiser: Denoiser,
        arpabet_dict,
        speaker_id: Optional[int] = None,
        gst_style: Optional[int] = None,
        device: Optional[torch.device] = None,
        out_path: Optional[str] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
) -> np.ndarray:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encode inputs
    text, style_input, speaker_ids, f0s = encode_input(
        text, reference_audio_path, arpabet_dict, mellotron, stft, hparams, speaker_id=speaker_id, device=device
    )
    encoded_input = (text, gst_style if gst_style is not None else style_input, speaker_ids, f0s)

    # In this case I only need to use the base Tacotron 2 for inference (Mellotron is not necessary)
    mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.infer(encoded_input)

    # Use vocoder to generate the raw audio signal
    audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]

    # Save waveform to file if path is provided
    if out_path is not None:
        write(out_path, hparams.sampling_rate, audio)

    # Plot generated Mel Spectrogram
    if plot:
        _plot_mel_f0_alignment(
            encoded_input[2].data.cpu().numpy()[0],
            mel_outputs_postnet.data.cpu().numpy()[0],
            encoded_input[-1].data.cpu().numpy()[0, 0],
            rhythm.data.cpu().numpy()[:, 0].T,
            output_path=plot_path
        )

    return audio
