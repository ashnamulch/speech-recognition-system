import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import soundfile as sf

class SpeechRecognitionModel:
    def __init__(self, model_name='facebook/wav2vec2-base', sample_rate=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.label_encoder = LabelEncoder()

    def load_audio(self, filepath):
        #waveform, sample_rate = torchaudio.load(filepath, format='wav')
        waveform, sample_rate = sf.read(filepath)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(torch.Tensor(waveform))
        return torch.Tensor(waveform).squeeze(0)
    
    def preprocess_audio(self, filepath):
        waveform = self.load_audio(filepath)
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors='pt').input_values
        with torch.no_grad():
            features = self.model(inputs).last_hidden_state
        return features.mean(dim=1).numpy().flatten()
    
    def prepare_data(self, audio_folder):
        audio_paths = []
        labels = []
        speakers = []

        for filename in os.listdir(audio_folder):
            if filename.endswith('.wav'):
                word, speaker, _ = filename[:-4].split('_')

                audio_paths.append(os.path.join(audio_folder, filename))
                labels.append(word)
                speakers.append(speaker)
        
        labels = self.label_encoder.fit_transform(labels)
        return np.array(audio_paths), labels
    
    def extract_features(self, audio_paths):
        features = [self.preprocess_audio(f) for f in audio_paths]
        return np.stack(features)
    
    def train_model(self, X_train, y_train):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        return clf
    
    def evaluate_model(self, clf, X_test, y_test):
        accuracy = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        return accuracy, conf_matrix, report
    
    def predict(self, clf, filepaths):
        for filepath in filepaths:
            features = self.preprocess_audio(filepath).reshape(1, -1)
            predicted_label = clf.predict(features)
            predicted_word = self.label_encoder.inverse_transform(predicted_label)
            print(f"Predicted transcription for {os.path.basename(filepath)}: {predicted_word[0]}")
    
def main():
    audio_folder = './audio_files'

    speech_recognition = SpeechRecognitionModel()

    audio_paths, labels = speech_recognition.prepare_data(audio_folder)

    X = speech_recognition.extract_features(audio_paths)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = speech_recognition.train_model(X_train, y_train)

    accuracy, conf_matrix, report = speech_recognition.evaluate_model(clf, X_test, y_test)

    print(f'Test accuracy: {accuracy * 100:.2f}%')
    print(conf_matrix)
    print(report)

    example_files = [
        './audio_files/apple_1_1.wav',
        './audio_files/balloon_2_1.wav',
        './audio_files/feather_3_1.wav',
        './audio_files/giraffe_4_1.wav',
        './audio_files/honey_1_2.wav',
        './audio_files/snow_2_3.wav'
    ]
    speech_recognition.predict(clf, example_files)

if __name__ == '__main__':
    main()
