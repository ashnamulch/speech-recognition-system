# Speech Recognition System Using Wav2Vec2

## Overview
This project implements a basic speech recognition system using Wav2Vec2 and logistic regression as the classifier. The system processes audio files and converts them into text using features extracted from Wav2Vec2. The model is trained and evaluated on a small dataset with multiple speakers and words. 

## Dataset
The dataset consists of 10 words recorded 5 times by 4 different speakers. Each word has 5 audio samples per speaker, resulting in 200 audio files in total. 

The words are:
- apple, balloon, cloud, feather, giraffe, honey, mountain, ocean, snow, tiger

The speakers are:
- age 21, female
- age 53, female
- age 58, male
- age 23, female

## Model Architecture
- **Wav2Vec2**: A pre-trained Wav2Vec2 model is used to extract features from raw audio files.
- **Logistic Regression**: A logistic regression classifier is trained on the extracted features to predict the spoken word.

## Dependencies
- Python 3.8+
- PyTorch
- Torchaudio
- Transformers (Hugging Face)
- Scikit-learn

To install the dependencies, run:
```bash
pip install -r requirements.txt
