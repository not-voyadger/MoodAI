# MoodAI – Facial Emotion Recognition with CNN

MoodAI is a Python project that uses a Convolutional Neural Network (CNN) to recognize emotions from facial images.

The trained model detects **7 emotion classes** via **FER2013**:  
`['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`

---

## Training Info

![alt text](https://github.com/not-voyadger/MoodAI/blob/develop/assets/screenshot1.PNG)

As seen on a screenshot, loss is decreasing with each epoch, which means that model is, in fact, learning.

After 10 epochs (~10 minutes) on one GPU, the test accuracy of recognition is reaching ~56%, which is expected for a simple CNN trained on limited data.  

---

## Installation

```bash
pip install torch torchvision
```

---

## Recent additions

- Continual training implemented. Now, after each training weights are being added to a model, not rewriting the whole model like a previous version.

---

## TODO

- Implement emotion detecting using webcam.
- Implement data augmentation to improve accuracy and robustness.

---


