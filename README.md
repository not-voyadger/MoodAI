# MoodAI – Facial Emotion Recognition with CNN

MoodAI is a Python project that uses a Convolutional Neural Network (CNN) to recognize emotions from facial images.

The trained model detects **7 emotion classes** via **FER2013**:  
`['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`

---

## Training Info

As seen on a screenshot, loss is decreasing with each epoch, which means that model is, in fact, learning:

![alt text](https://github.com/not-voyadger/MoodAI/blob/develop/assets/screenshot2.PNG)

![alt_text](https://github.com/not-voyadger/MoodAI/blob/develop/assets/screenshot3.PNG)

After 10 epochs (~30 minutes) on one GPU, the test accuracy of recognition is reaching ~60%, which is quite good for a simple CNN trained on limited data.  

---

## Installation

```bash
pip install torch torchvision
```

---

## Recent additions

- Continual training implemented. Now, after each training weights are being added to a model, not rewriting the whole model like a previous version.
- Model architecture modified, Residual blocks added to help the network preserve important features and train deeper layers more effectively. The input size of the first FC layer was corrected to match the flattened output of the convolutional layers. All new features resolved in an increase of accuracy from ~56% to ~60%.

---

## TODO

- Implement emotion detecting using webcam.
- Implement data augmentation to improve accuracy and robustness.

---


