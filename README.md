# KerasMNIST

Simple project to get keras (tensorflow as the backend) running on the 
MNIST dataset. The model being used is VGG 
(https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html). 
It is assumed that Keras and Tensorflow (as well as the other 
dependencies are installed and working correctly)

This program will train a VGG model to learn MNIST classification. This 
is just example code to output the accuracy and loss of the model. A 
simple `get_next()` batch function is implemented to get MNIST data 
batch by batch. figure_1.png is an example of the output at around 10 
epochs.

### Usage

```python
python3 main.py
```
