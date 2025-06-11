<div align="center">

<h1> Fine-grained emotional control of Text-to-Speech </h2>
<h3> LEARNING TO RANK INTER- AND INTRA-CLASS EMOTION INTENSITIES </h3>
<h4><i> Shijun Wang, Jón Guðnason, Damian Borth </i></h4>
<h4> ICASSP 2023 </h4>
</div>

<br>


Fine-grained emotional control for Text-to-Speech enables generation of speech with varying emotional intensities. This repository implements a ranking model that learns inter- and intra-class emotion strength and a FastSpeech2 based TTS system conditioned on those intensities. Preprocessing converts raw audio to features, aligns transcripts using Montreal Forced Aligner (MFA) and splits data for training. The EmoV-DB dataset is used, containing multiple speakers with several emotions each. Example scripts are provided for preparing data, training models and performing inference.

<br>

⚠️ This is unofficial implementation of the paper. Please refer the original paper.

<br>

## Environment
- Docker image: `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel`
- GPU: NVIDIA RTX 4060 (8GB VRAM)

<br>

## Setup
1. Clone this repository and install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the [EmoV-DB dataset](https://www.openslr.org/115/) and place it under `/workspace/data/EmoV-DB` (path can be changed in `parameter.yaml`).
3. Download the pretrained HiFi-GAN vocoder for LibriTTS (16kHz) to `/workspace/pretrained_models/tts-hifigan-libritts-16kHz`.

<br>

## Preprocessing
1. **Prepare MFA corpus**
   ```bash
   python rank_model/prepare_mfa.py
   ```
2. **Install Montreal Forced Aligner**
   ```bash
   conda create -n aligner -c conda-forge montreal-forced-aligner -y
   conda activate aligner
   mfa model download acoustic english_us_arpa
   wget -O /workspace/montreal_forced_aligner/librispeech-lexicon.txt \
         https://openslr.org/resources/11/librispeech-lexicon.txt
   mfa validate /workspace/montreal_forced_aligner/corpus \
              /workspace/montreal_forced_aligner/librispeech-lexicon.txt english_us_arpa
   mfa align /workspace/montreal_forced_aligner/corpus \
            /workspace/montreal_forced_aligner/librispeech-lexicon.txt english_us_arpa \
            /workspace/montreal_forced_aligner/aligned
   ```
   After alignment, return to the base environment:
   ```bash
   conda activate base
   ```
3. **Feature extraction**
   ```bash
   python rank_model/preprocess.py
   ```
4. **Prepare FastSpeech2 dataset splits**
   ```bash
   python fastspeech2/preprocess.py
   ```
<br>

## Training
Train the rank model and FastSpeech2 model sequentially:
```bash
PYTHONENV=. python rank_model/train.py
PYTHONENV=. python fastspeech2/train.py
```

<br>

## Inference
Generate speech using the trained models:
```bash
PYTHONENV=. python rank_model/inference.py
PYTHONENV=. python fastspeech2/inference.py
```