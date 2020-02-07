# Crystal Structure Classification

## Introduction
This work contributes a novel descriptor XRD-Edge and Multi-stream DenseNet (MSDN) for crystal structure classification.

We also provide the codes as follows:
  1) XRD-Edge descriptor (see [XRD-Edge](https://github.com/tiongleslie/crystal-structure-classification/tree/master/XRD-Edge))
  2) Pre-trained model of MSDN (see [MSDN](https://github.com/tiongleslie/crystal-structure-classification/tree/master/MSDN))

### Compatibility
We tested the codes with:
  1) Tensorflow-GPU 1.13.1 under Ubuntu OS 18.04 and Anaconda3 (Python 3.7)
  1) Tensorflow 1.13.1 under Windows 10 and Anaconda3 (Python 3.7)

### Requirements
  1) [Anaconda3](https://www.anaconda.com/distribution/#download-section)
  2) [TensorFlow-GPU 1.13.1 or Tensorflow 1.13.1](https://www.tensorflow.org/install/pip)
  3) [PIL](https://anaconda.org/anaconda/pillow)
  4) [NatSort](https://pypi.org/project/natsort/)
  5) [SciPy](https://anaconda.org/anaconda/scipy)
  6) [Matplotlib](https://anaconda.org/conda-forge/matplotlib)
  7) [Numpy 1.16.4](https://pypi.org/project/numpy/1.16.4/)
 
### Running sample code
#### XRD-Edge
- Run the code `test_XRD_Edge.py` in [XRD-Edge](https://github.com/tiongleslie/crystal-structure-classification/tree/master/XRD-Edge)
```shell
$ python test_XRD_Edge.py
```

#### Pre-trained model: MSDN
- Run the code `test_MSDN.py` in [MSDN](https://github.com/tiongleslie/crystal-structure-classification/tree/master/MSDN)
```shell
$ python test_MSDN.py --batch_size 16 --plot_sample 0
```

### Dataset
Please refer to [1] for the dataset information.

### Citation
Please cite the paper [1] if you are using our model in your research works.
  [1]
