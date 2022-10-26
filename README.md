# Mellotron API

API for [Mellotron](https://github.com/NVIDIA/mellotron).
This repository contains an installation guide and some utility functions to simplify the access to the model in this library.
All credits go to the original developers.

## Repository structure

This repository is organised into two main directories:

- `resources/` contains:
    - directories to host the Mellotron models;
    - directory to host the WaveGlow model.
- `tts_api/` package with the api.
- `mellotron/` submodule with Mellotron and WaveGlow code.

For further details on the available models, refer to the `README.md` in the `resources/` directory.

## Environment

To install all the required packages within an anaconda environment ans do a complete setup, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n ttsmellotron python=3.10 cudatoolkit=11.3
# Activate anaconda environment
conda activate ttsmellotron
# Install packages
conda install ...
# Download and initialise submodules
...
# Update ...
```

To add the directories to the Python path, you can add these lines to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/tts_mellotron_api/src/tts_mellotron_api
export PYTHONPATH=$PYTHONPATH:/path/to/tts_mellotron_api/mellotron
export PYTHONPATH=$PYTHONPATH:/path/to/tts_mellotron_api/mellotron/waveglow
```

## Example

Here follows a usage example:
```python
import torch
from tts_mellotron_api import ...


# Reference audio for voice (optional)
audio_path = 'path/to/audio.wav'
# Load model instances
...

# Syntehsise speech
synthesise_speech(
    "I am testing a neural network for speech synthesis.", 
    ...,
    device=torch.device('cpu'), 
    out_path='path/to/output.wav'
)
```
