# MoodAI – Facial Emotion Recognition with CNN

MoodAI is a Python project that uses a Convolutional Neural Network (CNN) to recognize emotions from facial images.

The trained model detects **7 emotion classes** via **FER2013**:  
`['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`

---

## Training Highlights

- Loss decreases with each epoch, showing that the model is learning.  
- After 10 epochs, the test accuracy reached ~56%, which is expected for a simple CNN trained on limited data.  
- Images are processed as **grayscale 48×48**, normalized to `[-1, 1]` for better learning.

---

## Installation

```bash
pip install torch torchvision
```

---

## TODO

- Implement **continual training**: load existing model weights before training to continue improving the model instead of starting from scratch.
- Implement data augmentation to improve accuracy and robustness.


