<div align="center">

<h1> Fine-grained emotional control of Text-to-Speech </h2>
<h3> LEARNING TO RANK INTER- AND INTRA-CLASS EMOTION INTENSITIES </h3>

</div>

<br>




# Rank Model

```
python prepare_mfa.py
```

```
# download speech dictionary
wget -O /workspace/montreal_forced_aligner/librispeech-lexicon.txt https://openslr.trmal.net/resources/11/librispeech-lexicon.txt 

# prepare environment for montreal forced aligner
conda create -n aligner -c conda-forge montreal-forced-aligner -y
conda activate aligner

# **important** please make sure to select `aligner` environment
mfa model download acoustic english_us_arpa
mfa validate /workspace/montreal_forced_aligner/corpus /workspace/montreal_forced_aligner/librispeech-lexicon.txt english_us_arpa
mfa align /workspace/montreal_forced_aligner/corpus /workspace/montreal_forced_aligner/librispeech-lexicon.txt english_us_arpa /workspace/montreal_forced_aligner/aligned
```

```
conda activate base
python preprocess.py
```

```
python train.py
```

```
python inference.py
```

# Fastspeech2

```
python preprocess.py

python train.py

python inference.py
```