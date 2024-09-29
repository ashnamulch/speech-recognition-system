# Speech Recognition System Using Wav2Vec2

## Overview
This project implements a basic speech recognition system using Wav2Vec2 and logistic regression as the classifier. The system processes audio files and converts them into text using features extracted from Wav2Vec2. The model is trained and evaluated on a small dataset with multiple speakers and words. 

## Dataset
The dataset consists of 10 words recorded 5 times by 6 different speakers. Each word has 5 audio samples per speaker, resulting in 300 audio files in total. 

The words are:
- apple, balloon, cloud, feather, giraffe, honey, mountain, ocean, snow, tiger

The speakers are:
- age 21, female
- age 53, female
- age 58, male
- age 23, female
- age 21, male
- age 21, male

## Model Architecture
- **Wav2Vec2**: A pre-trained Wav2Vec2 model is used to extract features from raw audio files.
- **Logistic Regression**: A logistic regression classifier is trained on the extracted features to predict the spoken word.

## Dependencies
- Python 3.8+
- PyTorch
- Torchaudio
- Transformers (Hugging Face)
- Scikit-learn

## How to run this program
1. Clone this repository and navigate into the project directory:
  ```bash
  git clone https://github.com/ashnamulch/speech-recognition-system.git
  cd speech-recognition-system
  ```

2. Ensure you have the necessary dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```

3. Run the program by executing the model.py script:
  ```bash
  python src/model.py
  ```

## Example Output
1. Accuracy score: Percentage of correctly clasified words.
```bash
Test accuracy: 93.33%
```

2. Confusion matrix: How often the model predicts each word correctly, as well as its mistakes. Each row represents the actual word, and each column represents the predicted word. The diagonal elements represent correct predictions, while off-diagonal elements show misclassifications.
```bash
[[8 0 0 0 0 0 0 0 0 0]
 [0 8 0 0 0 0 0 0 0 0]
 [0 0 5 0 0 0 0 0 0 0]
 [0 0 0 5 0 1 0 0 0 1]
 [0 0 0 0 5 0 0 0 0 0]
 [0 0 0 0 0 4 0 0 0 0]
 [0 0 0 0 0 0 2 0 0 0]
 [0 0 0 0 0 0 0 6 0 0]
 [0 0 0 0 0 0 1 0 7 0]
 [0 0 0 1 0 0 0 0 0 6]]
```
In this output, the model predicted 8 examples of the first word correctly, and none were misclassified as another word.

3. Classification report: Provides more details about the model's performance on each word.
   
   a. Precision: Percentage of correctly predicted instances out of the total predicted instances for a word.

   b. Recall: Percentage of correctly predicted instances out of the total actual instances of a word.
   
   c. F1-Score: A combination of precision and recall that balances both metrics.
```bash
              precision    recall  f1-score   support

       apple       1.00      1.00      1.00         8
     balloon       1.00      1.00      1.00         8
       cloud       1.00      1.00      1.00         5
     feather       0.83      0.71      0.77         7
     giraffe       1.00      1.00      1.00         5
       honey       0.80      1.00      0.89         4
    mountain       0.67      1.00      0.80         2
       ocean       1.00      1.00      1.00         6
        snow       1.00      0.88      0.93         8
       tiger       0.86      0.86      0.86         7

    accuracy                           0.93        60
   macro avg       0.92      0.94      0.92        60
weighted avg       0.94      0.93      0.93        60
```
In this output, precision for "apple" is 1.00, meaning 100% of instances predicted as "apple" were correct. Recall for "feather" is 0.71, meaning correctly identified 71% of all actual "feather" samples. The F1-Score gives an overall performance score for each word.

## Future Work
- Experimenting with larger datasets and more diverse speakers
- Fine-tuning the Wav2Vec2 model on domain-specific speech data
- Exploring other classification algorithms for better accuracy
