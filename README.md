# Pytorch Adversarial Autoencoders
Replicated the results from [this blog post](https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/) using PyTorch.

Using [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to view the trainging from [this repo.](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)

Autoencoders can be used to reduce dimensionality in the data.  This example uses the Encoder to fit the data (unsupervised step) and then uses the encoder representation as "features" to train the labels.

The result is not as good as using the raw features with a simple NN.  This example is designed to demonstrate the workflow for AAE and using that as features for a supervised step.

<br>

## Usage

#### 1. Install the dependencies
```bash
$ pip install -r requirements.txt
```

#### 2. Train the AAE model & supervised step
```bash 
$ python main_aae.py && python main.py
```

#### 3. Open TensorBoard to view training steps
To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ in your web browser.
```bash
$ tensorboard --logdir='./logs' --port=6006
```



