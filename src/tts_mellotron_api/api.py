from matplotlib import pyplot as plt


import numpy as np
import random
from scipy.io.wavfile import write
import librosa
import torch

from mellotron.hparams import create_hparams
from mellotron.model import Tacotron2, load_model
from mellotron.waveglow.denoiser import Denoiser
from mellotron.layers import TacotronSTFT
from mellotron.data_utils import TextMelCollate
from mellotron.text import cmudict, text_to_sequence
from mellotron.yin import compute_yin

from typing import Optional, Tuple


def load_tts(
        tts_model_checkpoint_path: str,
        device: Optional[torch.device] = None,
) -> Tuple[Tacotron2, object]:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hparams = create_hparams()
    mellotron = load_model(hparams).to(device).eval()
    mellotron.load_state_dict(torch.load(tts_model_checkpoint_path)['state_dict'])

    return mellotron, hparams


def load_vocoder(
        vocoder_model_checkpoint_path: str,
        device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Denoiser]:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    waveglow = torch.load(vocoder_model_checkpoint_path)['model'].to(device).eval()
    denoiser = Denoiser(waveglow).cuda().eval()

    return waveglow, denoiser


def load_arpabet_dict(dict_path: str):
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    arpabet_dict = cmudict.CMUDict(dict_path)

    return arpabet_dict


def _get_mel_spec(
        audio_path: str,
        hparams,
        device: Optional[torch.device] = None
) -> torch.FloatTensor:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/data_utils.py
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create STFT PyTorch layer
    stft = TacotronSTFT(
        hparams.filter_length,
        hparams.hop_length,
        hparams.win_length,
        hparams.n_mel_channels,
        hparams.sampling_rate,
        hparams.mel_fmin,
        hparams.mel_fmax
    )
    # Load audio file
    audio, sampling_rate = librosa.core.load(audio_path, sr=hparams.sampling_rate)
    # Convert to PyTorch format
    audio = torch.from_numpy(audio)
    # Normalize
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    # Compute Mel spectrogram using utility layer
    mel_spec = stft.mel_spectrogram(audio_norm)
    mel_spec = mel_spec.to(device)

    return mel_spec


def _get_f0(
        hparams,
        audio_path: Optional[str] = None,
        mel_spec: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None
) -> torch.FloatTensor:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/data_utils.py
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enforce either audio path or Mel spectrogram but not both
    assert not (audio_path is None and mel_spec is None) and not (audio_path is not None and mel_spec is not None)

    # Get input Mel Spectrogram
    mel_spec = mel_spec if mel_spec is not None else _get_mel_spec(audio_path, hparams, device=device)

    # Compute F0
    f0, harmonic_rates, argmins, times = compute_yin(
        mel_spec,
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
    f0 = f0[:, :mel_spec.shape[1]]

    return f0


@torch.no_grad()
def get_prosody_embedding(
        reference_audio_path: str,
        mellotron: Tacotron2,
        hparams,
        device: Optional[torch.device] = None
) ->torch.FloatTensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Mel spectrogram
    mel_spec = _get_mel_spec(reference_audio_path, hparams, device=device)
    # Compute prosody embedding via reference encoder
    prosody_embedding = mellotron.gst.encoder(mel_spec)

    return prosody_embedding


@torch.no_grad()
def get_gst(
        reference_audio_path: str,
        mellotron: Tacotron2,
        hparams,
        device: Optional[torch.device] = None
) -> torch.FloatTensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Mel spectrogram
    mel_spec = _get_mel_spec(reference_audio_path, hparams, device=device)

    ...


def encode_input(
        text: str,
        reference_audio_path: str,
        arpabet_dict,
        mellotron: Tacotron2,
        hparams,
        speaker_id: Optional[int] = None,
        device: Optional[torch.device] = None
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create
    data_collate = TextMelCollate(1)

    # Encode text
    text = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict), device=device)[None, :]
    # Compute Reference audio Mel spectrogram
    mel_spec = _get_mel_spec(reference_audio_path, hparams, device=device)
    # Compute pitch contour
    f0 = _get_f0(hparams, mel_spec=mel_spec.cpu().numpy(), device=device)
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
        mellotron: Tacotron2,
        hparams,
        waveglow,
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
        text, reference_audio_path, arpabet_dict, mellotron, hparams, speaker_id=speaker_id, device=device
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
