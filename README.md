# Mellotron API

API for [Mellotron](https://github.com/NVIDIA/mellotron).
This repository contains an installation guide and some utility functions to simplify the access to the model in this library.
All credits go to the original developers.

## Repository structure

This repository is organised into two main directories:

- `resources/` contains:
    - directory to host the Mellotron and Tacotron 2 models;
    - directory to host the WaveGlow model.
- `src/mellotron_api` package with the api.
- `mellotron/` submodule with Mellotron, WaveGlow, and Tacotron 2 code.
- `tmp/` contains temporary replacement files to have the code work properly (see environment section).

For further details on the available models, refer to the `README.md` in the `resources/` directory.

## Environment

To install all the required packages within an anaconda environment and do a complete setup, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n ttsmellotron python=3.10 cudatoolkit=11.3
# Activate anaconda environment
conda activate ttsmellotron
# Install packages
conda install pytorch=1.11.0 -c pytorch
conda install -c conda-forge scipy matplotlib librosa tensorflow music21 inflect tensorboard tensorboardx unidecode
conda install -c anaconda nltk pillow
pip install jamo
# Download and initialise submodules
# Mellotron
git submodule init; git submodule update
# WaveGlow inside Mellotron
cd mellotron/
git submodule init; git submodule update
cd ..
# Update code using old TensorFlow version with now deprecated interfaces
cp tmp/updated_hparams.py mellotron/hparams.py
cp tmp/updated_mellotron_model.py mellotron/model.py
cp tmp/updated_denoiser.py mellotron/waveglow/denoiser.py
cp tmp/updated_glow.py mellotron/waveglow/glow.py
```

To add the directories to the Python path, you can add these lines to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/tts_mellotron_api/src
export PYTHONPATH=$PYTHONPATH:/path/to/tts_mellotron_api/mellotron
export PYTHONPATH=$PYTHONPATH:/path/to/tts_mellotron_api/mellotron/waveglow
```

## Example

Here follows a usage example:
```python
import torch
from mellotron_api import load_tts, load_vocoder, load_arpabet_dict, synthesise_speech


# Reference audio for voice (optional)
audio_path = 'path/to/audio.wav'
# Load model instances
mellotron, stft, hparams = load_tts('resources/tts/mellotron/mellotron_libritts.pt')
waveglow, denoiser = load_vocoder('resources/vocoder/waveglow/waveglow_256channels_universal_v4.pt')
arpabet_dict = load_arpabet_dict('mellotron/data/cmu_dictionary')

# Syntehsise speech
synthesise_speech(
    "I am testing a neural network for speech synthesis.", 
    audio_path,
    mellotron,
    stft,
    hparams,
    waveglow,
    denoiser,
    arpabet_dict,
    device=torch.device('cpu'), 
    out_path='path/to/output.wav'
)
```
