# neural-style: Neural Style in Pytorch!

An implementation of the neural style in PyTorch! This notebook implements [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon Gatys, Alexander Ecker, and Matthias Bethge. 

This implementation is inspired by the implementations of:
* Anish Athalye [Neural Style in Tensorflow](https://github.com/anishathalye/neural-style),
* Justin Johnson [Neural Style in Torch](https://github.com/jcjohnson/neural-style), and
* ProGamerGov [Neural Style in PyTorch](https://github.com/ProGamerGov/neural-style-pt)

The [original caffe pretrained weights of VGG19](https://github.com/jcjohnson/pytorch-vgg) were used for this implementation, instead of the pretrained VGG19's in PyTorch's model zoo.

## Examples

### Janelle Monae + Starry Sky by Vincent van Gogh
![Janelle Monae](https://i.imgur.com/WWq6I1U.jpg)

### Andrew Y. Ng + Oil Painting of a Girl in Rain
![AndrewYNg](https://i.imgur.com/cO9YdZI.jpg)

### Style Transfer of Golden Bridge
![Golden Bridge](https://i.imgur.com/E6A4AmU.jpg)


### Data Files
* [Pre-trained VGG19 network weights](https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth) - put it in `models/` directory
* [torchvision](https://pytorch.org/) - `torchvision.models` contains the VGG19 model skeleton

## Requirements
`NOTE`: For `Google-Colab users` - All data files and dependencies can be installed by running the uppermost cell of the notebook!

### Dependecies
* [PyTorch](https://pytorch.org/)
* [NumPy](https://www.scipy.org/install.html)
* [Jupyter](http://jupyter.org/install)
* [opencv2](https://matplotlib.org/users/installing.html)
* [Copy](https://docs.python.org/3/library/copy.html)

## Todo!
* Multiple Style blending
* High-res Style Transfer
* Color-preserving Style Transfer