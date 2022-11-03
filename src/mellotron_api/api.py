from matplotlib import pyplot as plt

import numpy as np
import random
from scipy.io.wavfile import write
import librosa
import torch

# Mellotron
from mellotron.hparams import create_hparams as create_mellotron_haprams, AttrDict as MellotronAttrDict
from mellotron.model import Tacotron2 as Mellotron, load_model as load_mellotron_model
from mellotron.layers import TacotronSTFT as MellotronSTFT
from mellotron.data_utils import TextMelCollate
from mellotron.text import (
    cmudict,
    text_to_sequence as text_to_sequence_mellotron,
    sequence_to_text as sequence_to_text_mellotron
)
from mellotron.yin import compute_yin
# WaveGlow
from waveglow.glow import WaveGlow
from waveglow.denoiser import Denoiser
# Tacotron 2
from tacotron2.hparams import create_hparams as create_tacotron2_haprams, AttrDict as Tacotron2AttrDict
from tacotron2.model import Tacotron2
from tacotron2.train import load_model as load_tacotron2_model
from tacotron2.layers import TacotronSTFT
from tacotron2.text import text_to_sequence as text_to_sequence_tacotron2
from tacotron2.audio_processing import griffin_lim

from typing import Optional, Tuple, List, Union, Literal


def _load_mellotron(
        tts_model_checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[Mellotron, MellotronSTFT, MellotronAttrDict]:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load hyper-parameters
    hparams: MellotronAttrDict = create_mellotron_haprams()
    # Create STFT PyTorch layer
    stft = MellotronSTFT(
        filter_length=hparams.filter_length,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        n_mel_channels=hparams.n_mel_channels,
        sampling_rate=hparams.sampling_rate,
        mel_fmin=hparams.mel_fmin,
        mel_fmax=hparams.mel_fmax
    )
    # Create Mellotron TTS instance and load weights
    mellotron: Mellotron = load_mellotron_model(hparams).eval()
    mellotron.load_state_dict(torch.load(tts_model_checkpoint_path, map_location=device)['state_dict'])

    return mellotron, stft, hparams


def _load_tacotron2(
        tts_model_checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[Tacotron2, TacotronSTFT, Tacotron2AttrDict]:
    # See:
    # https://github.com/NVIDIA/tacotron2/blob/master/inference.ipynb
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load hyper-parameters
    hparams: Tacotron2AttrDict = create_tacotron2_haprams()
    # Create STFT PyTorch layer
    stft = TacotronSTFT(
        filter_length=hparams.filter_length,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        sampling_rate=hparams.sampling_rate
    )
    # Create Tacotron 2 TTS instance and load weights
    tacotron2: Tacotron2 = load_tacotron2_model(hparams).eval()
    tacotron2.load_state_dict(torch.load(tts_model_checkpoint_path, map_location=device)['state_dict'])

    return tacotron2, stft, hparams


def load_tts(
        tts_model_checkpoint_path: str,
        model: Literal['mellotron', 'tacotron2'] = 'mellotron',
        device: Optional[torch.device] = None
) -> Union[Tuple[Mellotron, MellotronSTFT, MellotronAttrDict], Tuple[Tacotron2, TacotronSTFT, Tacotron2AttrDict]]:
    # Forward  call to proper TTS loader
    if model == 'mellotron':
        return _load_mellotron(tts_model_checkpoint_path, device=device)
    elif model == 'tacotron2':
        return _load_tacotron2(tts_model_checkpoint_path, device=device)
    else:
        raise ValueError(f'Unsupported model type: {model}')


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
        hparams: MellotronAttrDict,
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
    mel_spec = stft.mel_spectrogram(audio_norm).squeeze()
    mel_spec = mel_spec.to(device)

    return mel_spec


def _get_f0(
        hparams: MellotronAttrDict,
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


def _get_speaker_id(
        speaker_id: Optional[int] = None, n_speakers: Optional[int] = None, device: Optional[torch.device] = None
) -> torch.tensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If speaker ID is none sample one randomly from model
    if speaker_id is None:
        speaker_id = random.randint(0, n_speakers - 1)
    speaker_id = torch.tensor(speaker_id, device=device)

    return speaker_id


def _get_style_input(
        style_input: torch.tensor,
        gst_style_id: Optional[int] = None,
        gst_style_scores: Optional[List[float]] = None,
        gst_head_style_scores: Optional[List[List[float]]] = None,
        gst_style_embedding: Optional[List[float]] = None,
        prosody_embedding: Optional[List[float]] = None,
        device: Optional[torch.device] = None
) -> torch.tensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select style control approach
    if gst_style_id is not None:
        return gst_style_id
    elif gst_style_scores is not None:
        return torch.tensor(gst_style_scores, device=device).unsqueeze(0)
    elif gst_head_style_scores is not None:
        return torch.tensor(gst_style_scores, device=device).unsqueeze(1).unsqueeze(1)
    elif gst_style_embedding is not None:
        return torch.tensor(gst_style_embedding, device=device).unsqueeze(0)
    elif prosody_embedding is not None:
        return torch.tensor(prosody_embedding, device=device).unsqueeze(0)
    else:
        return style_input


def _get_length_from_alignment(alignment: np.ndarray) -> int:
    # Compute alignment point for each time stamp
    tmp = np.argmax(alignment, axis=1)
    # Get points where alignment goes backwards after end
    candidates = np.where((tmp[:-1] == alignment.shape[-1] - 1) & (tmp[:-1] > tmp[1:]))[0]

    if len(candidates) > 0:
        tgt_len = candidates[0] + 1
    else:
        tgt_len = -1

    return tgt_len


@torch.no_grad()
def get_prosody_embedding(
        reference_audio_path: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: MellotronAttrDict,
        device: Optional[torch.device] = None,
        out_format: Literal['python', 'numpy', 'pytorch'] = 'numpy'
) -> Union[torch.tensor, np.ndarray, List[float]]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Mel spectrogram
    mel_spec = _get_mel_spec(stft, hparams, audio_path=reference_audio_path, device=device).unsqueeze(0)
    # Compute prosody embedding via reference encoder
    prosody_embedding = mellotron.gst.encoder(mel_spec)

    # Convert output to desired format and return
    if out_format == 'pytorch':
        return prosody_embedding.squeeze().cpu()
    elif out_format == 'numpy':
        return prosody_embedding.squeeze().cpu().numpy()
    elif out_format == 'python':
        return prosody_embedding.squeeze().cpu().tolist()
    else:
        raise ValueError(f'Unsupported output format: {out_format}')


@torch.no_grad()
def get_gst_scores(
        reference_audio_path: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: MellotronAttrDict,
        device: Optional[torch.device] = None,
        out_format: Literal['python', 'numpy', 'pytorch'] = 'numpy'
) -> Union[torch.tensor, np.ndarray, List[float]]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load prosody embedding
    prosody_embedding = get_prosody_embedding(
        reference_audio_path, mellotron, stft, hparams, device=device, out_format='pytorch'
    ).to(device)
    # Prepare attention inputs
    tgt = prosody_embedding.unsqueeze(0).unsqueeze(1)  # Used for query
    src = torch.tanh(mellotron.gst.stl.embed).unsqueeze(0).expand(1, -1, -1)  # Used for keys and values
    # Compute query, keys and values
    query = mellotron.gst.stl.attention.W_query(tgt)
    keys = mellotron.gst.stl.attention.W_key(src)
    split_size = mellotron.gst.stl.attention.num_units // mellotron.gst.stl.attention.num_heads
    query = torch.stack(torch.split(query, split_size, dim=2), dim=0)
    keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
    # Compute the scores
    gst_scores = torch.matmul(query, keys.transpose(2, 3))
    gst_scores = gst_scores / (mellotron.gst.stl.attention.key_dim ** 0.5)
    gst_scores = torch.softmax(gst_scores, dim=3)

    # Convert output to desired format and return
    if out_format == 'pytorch':
        return gst_scores.squeeze().cpu()
    elif out_format == 'numpy':
        return gst_scores.squeeze().cpu().numpy()
    elif out_format == 'python':
        return gst_scores.squeeze().cpu().tolist()
    else:
        raise ValueError(f'Unsupported output format: {out_format}')


@torch.no_grad()
def get_gst_embeddings(
        reference_audio_path: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: MellotronAttrDict,
        device: Optional[torch.device] = None,
        out_format: Literal['python', 'numpy', 'pytorch'] = 'numpy'
) -> Union[torch.tensor, np.ndarray, List[float]]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Mel spectrogram
    mel_spec = _get_mel_spec(stft, hparams, audio_path=reference_audio_path, device=device).unsqueeze(0)
    # Use attention layer to compute the GST embeddings
    gst_embeddings = mellotron.gst(mel_spec)

    # Convert output to desired format and return
    if out_format == 'pytorch':
        return gst_embeddings.squeeze().cpu()
    elif out_format == 'numpy':
        return gst_embeddings.squeeze().cpu().numpy()
    elif out_format == 'python':
        return gst_embeddings.squeeze().cpu().tolist()
    else:
        raise ValueError(f'Unsupported output format: {out_format}')


def _encode_input_mellotron(
        text: str,
        arpabet_dict,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: MellotronAttrDict,
        reference_audio_path: Optional[str] = None,
        tacotron2: Optional[Tacotron2] = None,
        tacotron2_stft: Optional[TacotronSTFT] = None,
        tacotron2_hparams: Optional[Tacotron2AttrDict] = None,
        waveglow: Optional[WaveGlow] = None,
        denoiser: Optional[Denoiser] = None,
        speaker_id: Optional[int] = None,
        gst_style_id: Optional[int] = None,
        gst_style_scores: Optional[List[float]] = None,
        gst_head_style_scores: Optional[List[List[float]]] = None,
        gst_style_embedding: Optional[List[float]] = None,
        prosody_embedding: Optional[List[float]] = None,
        inference_mode: Literal['mellotron', 'tacotron2'] = 'mellotron',
        device: Optional[torch.device] = None
) -> Union[
    Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor],
    Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor]
]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create
    data_collate = TextMelCollate(1)

    # Encode text
    text_ids = torch.LongTensor(text_to_sequence_mellotron(text, hparams.text_cleaners, arpabet_dict), device=device)

    # Encode acoustic features
    # Extract acoustic features from reference audio if provided
    if reference_audio_path is not None:
        # Encode audio
        audio, _ = librosa.core.load(reference_audio_path, sr=hparams.sampling_rate)
        # Compute Reference audio Mel spectrogram
        mel_spec = _get_mel_spec(stft, hparams, audio=audio, device=device)
        # Dummy rhythm variable
        rhythm = None
    # Synthesise speech from Tacotron 2 if reference is not provided
    else:
        # Generate Mel spectrogram and alignment
        _, mel_spec, _, rhythm = _synthesise_speech_tacotron2(
            sequence_to_text_mellotron(text_to_sequence_mellotron(text, hparams.text_cleaners, arpabet_dict)),
            tacotron2,
            tacotron2_hparams,
            device
        )
        # Use vocoder or Griffin-Limm algorithm to reconstruct original audio
        audio = _synthesise_raw_audio(mel_spec, waveglow=waveglow, denoiser=denoiser, stft=tacotron2_stft)
        # Remove dummy batch dimension
        mel_spec = mel_spec.squeeze(0)
        audio = audio.squeeze(0)
    # Compute pitch contour
    f0 = _get_f0(hparams, audio=audio, mel_spec=mel_spec.cpu().numpy(), device=device)

    # Encode speaker ID
    speaker_id = _get_speaker_id(
        speaker_id=speaker_id, n_speakers=mellotron.speaker_embedding.num_embeddings, device=device
    )

    # Apply collate function
    (text_ids, text_id_lengths, mel_style_input, max_len, output_lengths, speaker_ids, f0s), _ = mellotron.parse_batch(
        data_collate([(text_ids, mel_spec, speaker_id, f0)])
    )

    # Encode GST
    style_input = _get_style_input(
        mel_style_input,
        gst_style_id=gst_style_id,
        gst_style_scores=gst_style_scores,
        gst_head_style_scores=gst_head_style_scores,
        gst_style_embedding=gst_style_embedding,
        prosody_embedding=prosody_embedding,
        device=device
    )

    # Check for rhythm data in case of Mellotron inference
    if inference_mode == 'mellotron':
        # Generate rhythm data if not available
        if rhythm is None:
            # Use Mellotron's underlying Tacotron 2 model to get the alignment (rhythm)
            *_, rhythm = mellotron.forward(
                (text_ids, text_id_lengths, mel_style_input, max_len, output_lengths, speaker_ids, f0s)
            )
        rhythm = rhythm.permute(1, 0, 2)


    # Return encoded input depending on the chosen inference mode
    if inference_mode == 'tacotron2':  # Use underlying generative Tacotron 2 model
        return text_ids, style_input, speaker_ids, f0s
    elif inference_mode == 'mellotron':  # Use Mellotron model to postprocess reference spectrogram
        return text_ids, style_input, speaker_ids, f0s, rhythm
    else:
        raise ValueError(f'Unsupported inference mode: {inference_mode}')


def _plot_mel_f0_alignment(
        mel_source, mel_outputs_postnet, f0s, alignments, figsize=(16, 16), output_path: Optional = None
):
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto', origin='lower', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto', origin='lower', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto', origin='lower', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)


def _synthesise_speech_mellotron(
        text: str,
        mellotron: Mellotron,
        stft: MellotronSTFT,
        hparams: MellotronAttrDict,
        arpabet_dict,
        inference_mode: Literal['mellotron', 'tacotron2'] = 'mellotron',
        len_check: bool = False,
        len_src: Literal['pitch', 'alignment'] = 'pitch',
        plot: bool = False,
        plot_path: Optional[str] = None,
        **kwargs
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    # See:
    # https://github.com/NVIDIA/mellotron/blob/master/inference.ipynb

    # Encode inputs
    encoded_input = _encode_input_mellotron(
        text, arpabet_dict, mellotron, stft, hparams, inference_mode=inference_mode, **kwargs
    )

    # Generate spectrogram depending on the selected inference mode
    if inference_mode == 'tacotron2':  # Use underlying generative Tacotron 2 model
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.inference(encoded_input)
    elif inference_mode == 'mellotron':  # Use Mellotron model to postprocess reference spectrogram
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.inference_noattention(encoded_input)
    else:
        raise ValueError(f'Unsupported inference mode: {inference_mode}')

    # TODO add rhythm based post processing to resample the pitch (call function recursively after scaling)
    # Additional check to make sure that the output matches the length of the reference audio pitch contour
    if len_check:
        # Get target length
        if len_src == 'pitch':
            tgt_len = encoded_input[3].size(-1)
        elif len_src == 'alignment':
            tgt_len = _get_length_from_alignment(rhythm.cpu().squeeze(0).numpy())
        else:
            raise ValueError(f'Unsupported targt length source: {len_src}')
        # Check on Mel spectrogram
        mel_outputs = mel_outputs[:, :, :tgt_len]
        # Check on Postnet Mel spectrogram
        mel_outputs_postnet = mel_outputs_postnet[:, :, :tgt_len]
        # Gate outputs
        gate_outputs = gate_outputs[:, :tgt_len]
        # Check on alignment (rhythm) # NOTE it may not be necessary
        rhythm = rhythm[:, :tgt_len]

    # Plot generated Mel Spectrogram
    if plot:
        try:
            _plot_mel_f0_alignment(
                encoded_input[1].data.cpu().numpy()[0],
                mel_outputs_postnet.data.cpu().numpy()[0],
                encoded_input[3].data.cpu().numpy()[0, 0],
                rhythm.data.cpu().numpy()[0].T,
                output_path=plot_path
            )
        except:  # TODO check for correct error
            raise Warning("An error occurred while plotting, skipping...")

    return mel_outputs, mel_outputs_postnet, gate_outputs, rhythm


def _encode_input_tacotron2(
        text: str,
        hparams: Tacotron2AttrDict,
        device: Optional[torch.device] = None
) -> torch.tensor:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use utility function to convert input string to NumPy array
    text_ids = np.array(text_to_sequence_tacotron2(text, hparams.text_cleaners))[None, :]
    # Convert to PyTorch  format
    text_ids = torch.from_numpy(text_ids).to(device).long()

    return text_ids


def _synthesise_speech_tacotron2(
        text: str,
        tacotron2: Tacotron2,
        hparams: Tacotron2AttrDict,
        device: Optional[torch.device] = None,
        plot: bool = False,
        plot_path: Optional[str] = None
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert input text string to PyTorch tensor
    encoded_input = _encode_input_tacotron2(text, hparams, device=device)

    # Process text input with Tacotron 2 and retrieve generated Mel spectrogram
    mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = tacotron2.inference(encoded_input)

    # Plot generated Mel Spectrogram
    if plot:
        raise NotImplementedError()

    return mel_outputs, mel_outputs_postnet, gate_outputs, rhythm


def _get_spec(mel_spec: torch.tensor, stft: Union[MellotronSTFT, TacotronSTFT]):
    # Invert Mel spectrogram into linear spectrogram
    mel_decompress = stft.spectral_de_normalize(mel_spec)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    return spec_from_mel


def _synthesise_raw_audio(
        mel_spec: torch.tensor,
        waveglow: Optional[WaveGlow] = None,
        denoiser: Optional[Denoiser] = None,
        stft: Optional[Union[MellotronSTFT, TacotronSTFT]] = None
) -> torch.tensor:
    # If the vocoder is available use is to synthesise audio
    if waveglow is not None:
        audio = waveglow.infer(mel_spec, sigma=0.8)
    # Else use Griffin-Limm algorithm
    else:
        audio = griffin_lim(_get_spec(mel_spec, stft)[:, :, :-1], stft.stft_fn, 60)

    # Apply denoising if denoiser is available
    if denoiser is not None:
        audio = denoiser(audio, 0.01)[:, 0]

    return audio


@torch.no_grad()
def synthesise_speech(
        text: str,
        tts_model: Union[Mellotron, Tacotron2],
        hparams: Union[MellotronAttrDict, Tacotron2AttrDict],
        stft: Optional[Union[MellotronSTFT, TacotronSTFT]] = None,
        arpabet_dict: Optional = None,
        waveglow: Optional[WaveGlow] = None,
        denoiser: Optional[Denoiser] = None,
        out_path: Optional[str] = None,
        **kwargs
) -> np.ndarray:
    # Synthesise Mel spectrogram
    if isinstance(tts_model, Mellotron):
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = _synthesise_speech_mellotron(
            text,
            tts_model,
            stft,
            hparams,
            arpabet_dict,
            waveglow=waveglow,
            denoiser=denoiser,
            **kwargs
        )
    elif isinstance(tts_model, Tacotron2):
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = _synthesise_speech_tacotron2(
            text,
            tts_model,
            hparams,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported TTS model: {type(tts_model)}")

    # Convert Mel spectrogram into raw audio
    audio = _synthesise_raw_audio(mel_outputs_postnet, waveglow=waveglow, denoiser=denoiser, stft=stft)

    # Finally recover audio track on CPU as NumPy ndarray
    audio = audio.cpu().numpy()[0]

    # Save waveform to file if path is provided
    if out_path is not None:
        write(out_path, hparams.sampling_rate, audio)

    return audio
