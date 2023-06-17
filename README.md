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
Epoch 1 loss: 0.147
Epoch 2 loss: 0.089
Epoch 3 loss: 0.073
Epoch 4 loss: 0.061
Epoch 5 loss: 0.051
Epoch 6 loss: 0.045
Epoch 7 loss: 0.035
Epoch 8 loss: 0.033
Epoch 9 loss: 0.030
Epoch 10 loss: 0.026
Validation accuracy: 97.378%
Test accuracy: 97.239%
F1 Score: 0.948%
Precision: 0.959%
Recall: 0.938%