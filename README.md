# Fire and Smoke Detection Using Computer Vision

This project is a part of the BBM416 course, conducted by the Computer Engineering Department of Hacettepe University. The main objective of this project is to detect fires and smoke using computer vision techniques.

![computer_vision_video_3](https://github.com/davutkulaksiz/Computer-Vision-Term-Project/assets/58954450/540896cf-9d90-48c7-a7a0-517992d96f82)

## 🚀 Problem Statement

Fire and smoke breakouts can be a major risk to public safety, and early detection is crucial to minimize the damage caused by fires. Traditional fire detection methods can have limitations in terms of accuracy and speed. Therefore, early detection using computer vision techniques can be helpful.

## 💡 Dataset

We will be using the following dataset to train and test our fire and smoke detection model:

| Dataset Name | Description | Number of Images |
| --- | --- | --- |
| [**D-Fire Dataset**](https://github.com/gaiasd/DFireDataset) | Contains more than 21,000 images of fire and smoke occurrences, including flames, smoke clouds, and fire sources, as well as non-fire images. | 21,529 |

## 🔍 Solution Strategy

To identify fires and smoke in the images, our suggested solution involves the usage of computer vision techniques like image classification and object detection.

- We will use the D-Fire dataset to train a Convolutional Neural Network (CNN) model to classify images as either fire or smoke and perform object detection on the images.
- We will evaluate our model's performance using standard metrics like precision, recall, and F1-score.

## 👥 Group Members

- Deniz Gönenç
- M. Davut Kulaksız
- Hikmet Güner

## Classification:

| Epoch | Loss | Validation Accuracy | Test Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|
| 1 | 0.147 | 96.692% | 96.555% | 0.943% | 0.954% | 0.932% |
| 2 | 0.089 | 97.087% | 96.950% | 0.948% | 0.960% | 0.936% |
| 3 | 0.073 | 97.283% | 97.146% | 0.950% | 0.962% | 0.938% |
| 4 | 0.061 | 97.479% | 97.342% | 0.952% | 0.964% | 0.940% |
| 5 | 0.051 | 97.675% | 97.538% | 0.954% | 0.966% | 0.942% |
| 6 | 0.045 | 97.871% | 97.734% | 0.956% | 0.968% | 0.944% |
| 7 | 0.035 | 98.067% | 97.930% | 0.958% | 0.970% | 0.946% |
| 8 | 0.033 | 98.263% | 98.126% | 0.959% | 0.971% | 0.947% |
| 9 | 0.030 | 98.459% | 98.322% | 0.961% | 0.973% | 0.949% |
| 10 | 0.026 | 98.655% | 98.518% | 0.962% | 0.974% | 0.950% |
