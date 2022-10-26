# Models

This directory is used to host the pre-trained models (TTS and vocoder).
All credits to the original authors and contributors that trained the models (see links below).

Voice synthesis:
- Spectrogram generation:
  - [Mellotron](https://doi.org/10.1109/ICASSP40776.2020.9054556):
    - Trained on [LibriTTS](https://openslr.org/60/) ([weights checkpoint](https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI));
    - Trained on [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) ([weights checkpoint](https://drive.google.com/open?id=1UwDARlUl8JvB2xSuyMFHFsIWELVpgQD4)).   
- Vocoder:
  - [WaveGlow](https://doi.org/10.1109/ICASSP.2019.8683143) ([weights checkpoint](https://drive.google.com/open?id=1okuUstGoBe_qZ4qUEF8CcwEugHP7GM_b))
  
Please refer to the [Mellotorn](https://github.com/NVIDIA/mellotron) and [Waveglow](https://github.com/NVIDIA/waveglow) repositories by [NVIDIA](https://www.nvidia.com) for further details.
For simplicity, we provide a separate zip file with all the model checkpoints necessary to speech synthesis ([link]()).

Directory structure:
```
 |- resources/
    |- mellotron/
      |- mellotron_ljs.pt
      |- mellotron_libritts.pt
    |- waveglow/
      |- waveglow_256channels_universal_v4.pt
```
