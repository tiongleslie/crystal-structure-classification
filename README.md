# Crystal Structure Classification

## Introduction
This work contributes a novel descriptor XRD-Edge and Multi-stream DenseNet (MSDN) for crystal structure classification.
Please cite the paper [1] if you are using our model in your research works.

We also provide the codes as follows:
  1) XRD-Edge descriptor (see [XRD-Edge](https://github.com/tiongleslie/crystal-structure-classification/tree/master/XRD-Edge))
  2) Pre-trained model of MSDN (see [MSDN](https://github.com/tiongleslie/crystal-structure-classification/tree/master/MSDN))

### Compatibility
The codes are tested using Tensorflow-GPU r1.13 under Ubuntu OS 18.04 and Anaconda3 (Python 3.7) environment.

### Requirements
  1) [Anaconda3](https://www.anaconda.com/distribution/#download-section)
  2) [TensorFlow-GPU 1.13](https://www.tensorflow.org/install/pip)
  3) [PIL](https://anaconda.org/anaconda/pillow)
  4) [NatSort](https://pypi.org/project/natsort/)
  5) [SciPy](https://anaconda.org/anaconda/scipy)
  6) [Matplotlib](https://anaconda.org/conda-forge/matplotlib)
  7) [Numpy 1.17.5](https://pypi.org/project/numpy/1.17.5/)
 
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

### Paper Citation
  [1]
