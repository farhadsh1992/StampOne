
# StampOne: Addressing Frequency Balance in Printer-proof Steganography [Paper](https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Shadmand_StampOne_Addressing_Frequency_Balance_in_Printer-proof_Steganography_CVPRW_2024_paper.html)

![alt text](https://github.com/farhadsh1992/StampOne.git/Images/encoded_samples)

## Introduction:

Robust steganography and invisible watermarking techniques in printed images are crucial for anti-counterfeiting systems within the multimedia industry for copyright protection security of documents (e.g. passports) and brand protection graphic elements. Conventional steganography models mainly designed for digital non-lossy media encounter challenges in recovering messages from images degraded by printing and scanning or social media compression particularly due to limitations associated with utilizing image regions characterized by the lowest and highest frequencies. In this paper we introduce StampOne a novel printer-proof steganography model utilizing Generative Adversarial Networks (GANs). StampOne ensures balanced frequency density between encoder and decoder inputs reducing disparities between original and encoded images. Our method through integration with diverse U-shape networks (image-to-image) emphasizes the significance of frequency domain analysis in robust steganography. It facilitates the development of robust steganography models capable of withstanding diverse noise types including JPEG compression contrast variations brightness fluctuations aliasing blurring and Gaussian noises. It surpasses previous models in both quality of encoded images and printer-proof capabilities. 

<hr>

## Installation
Python ```3.9.13```,  keras 3.1 and TensorFlow ```2.16.1``` are used in this implementation.

It is recommended to create conda env from our provided environment.yml:
```terminal
conda env create -f environment.yml
conda activate StampOne_keras3
```

Or you can install neccessary libraries as follows:

```terminal
conda create -n StampOne_keras3 python==3.9.13
conda activate StampOne_keras3
pip install -r requirements.txt
```

### Pre-trained model
For downlaod pretrained encoder and decoder (tflite format), contact us ```Farhadsh1992@gmail.com```

<hr>

## Detector Models

A. FaceDetection

Based on [PRNet](https://github.com/yfeng95/PRNet), this step includes detecting, cropping, and aligning faces.
B. ObjectDetection

Based on [YoloV5](https://github.com/LongxingTan/tfyolo)  for object detection, this step involves detecting and cropping the largest object in the background.
C. Border

Using OpenCV to add a border with a specific color around the encoded part. This color should be calibrated for the sensor separately. More details can be found [here](https://www.tutorialspoint.com/color-identification-in-images-using-python-and-opencv).
D. QRCode

Adding a QR code pattern in the corner of the image to facilitate the detection of the encoded part of the image.



<hr>

## Encode the original images
Preprocessing the inputs of the encoder network by reshaping the 256-bit 
binary sequences into a 16×1616×16 2D matrix in grayscale image format. This 2D message 
is then converted to a 3D RGB image format. 
Both the message and the cover image undergo gradient and wavelet operations.
The wavelet transform is applied to achieve dimensions of 16×16×1516×16×15 for 
the message and 256×256×15256×256×15 for the original image. 
Subsequently, the "Depthwise" layer is employed to assign distinct weights to 
each of the discrete wavelet transform (DWT) sub-bands. The highlighted message 
in the wavelet domain is then forwarded to the Message Preparation Network (MPN).

The pre-trained U-shape network, AttentionVNet, is available for use.

```terminal
bashFile/run_encoder.sh
```

<hr>

## Decode the encoded images
The gradient and wavelet transformations of the encoded images are processed 
through the "Depthwise" layer and the Spatial Transformer Network (STN) [23]. 
In the "Depthwise" layer, each channel of the image frequency wavelet is assigned 
a specific weight to emphasize the high-frequency components of the encoded image.
The STN is utilized to prevent warping and rotation when printing and capturing 
encoded images using a camera sensor.
The pre-trained U-shape network, AttentionVNet, is available for use.

```terminal
bashFile/run_decoder.sh
```

<hr>

## Evaluation  (metric)

#### Fréchet inception distance (FID) 
Fréchet Inception Distance (FID) is a metric used to quantify the realism and diversity of images 
generated by generative adversarial networks (GANs). Realism implies that the generated images, 
such as those of people, closely resemble real images. Diversity indicates that the generated 
images are sufficiently distinct from the originals, making them interesting and novel.


```terminal
bashFile/
```



### Perceptual Similarity and Diversity Metric (PSDM)
In addition to the widely used Fréchet Inception Distance (FID), 
which quantifies the realism and diversity of images generated by 
generative adversarial networks (GANs), we propose a new metric, 
the Perceptual Similarity and Diversity Metric (PSDM).

PSDM is designed to evaluate the quality of generated images 
by considering both perceptual similarity and diversity.

```terminal
bashFile/
```

#### Color Histogram (ColorHisto) or  HistoGan 
[HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms](link-gitbub: https://github.com/mahmoudnafifi/HistoGAN)

HistoGan leverages color histograms due to their intuitive representation of image color, 
which remains independent of domain-specific semantics. The color histogram is based on 
Log-Chroma space and calculates the Euclidean norm of the histogram 
features (H) between encoded and original images.

To quantify the color histogram between encoded and original images, the following steps are performed:

```terminal
bashFile/measure_ColorHisto.sh
```


#### Learned Perceptual Image Patch Similarity [(LPIPS)](https://github.com/richzhang/PerceptualSimilarity)
LPIPS uses a pretrained pyramid network to extract image features from different layers, and the average of these features is used tomeasure perceptual differences

Below is a code snippet to quantify the Learned Perceptual Image Patch Similarity (LPIPS) between encoded and original images using the LPIPS library in Python:

```terminal
bashFile/measure_LPIPS.sh
```


#### Structural Similarity (SSIM)
SSIM index is computed for the image with respect to the reference image. The reference image is usually needs to be of perfect quality.


```terminal
bashFile/measure_SSIM.sh
```

#### peak signal-to-noise ratio (PSNR) 
PSNR, is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation:

```terminal
bashFile/measure_PSNR.sh
```

<hr>

## Noise Simulation
If the paper "Noise simulation for the improvement of training 
deep neural network for printer-proof steganography"
has github page. I will refrence it. if not I make a file for that.

<hr>

## Dataset 
To perform our training experiments, we utilized subsets of two main datasets:

- [COCO Dataset](https://cocodataset.org/#home): Consisting of approximately 123,000 images.
- [DeepFashion Dataset](https://chatgpt.com/c/e8a5f90c-bb00-437b-bd37-d2dda4e93300): Consisting of approximately 800,000 images.

For our testing experiments, we utilized the following datasets:

- [BSDS500](https://chatgpt.com/c/e8a5f90c-bb00-437b-bd37-d2dda4e93300):A benchmark dataset for image segmentation, consisting of 500 natural images divided into training, validation, and test sets. This dataset is widely used for evaluating image processing and computer vision algorithms.
- [Urban](https://chatgpt.com/c/e8a5f90c-bb00-437b-bd37-d2dda4e93300): A dataset comprising high-resolution images of urban scenes, including buildings, streets, and various urban structures. This dataset is useful for tasks related to urban scene understanding and analysis.
- [VGGFace2](https://chatgpt.com/c/e8a5f90c-bb00-437b-bd37-d2dda4e93300): A large-scale face recognition dataset containing images of 9,131 subjects, with an average of 362.6 images per subject. The images exhibit large variations in pose, age, illumination, and background, making it ideal for testing face recognition algorithms.

These datasets provided a comprehensive basis for both training and evaluating the performance of our models.
<hr>

## Results

<hr>

## Acknowledgments
We extend our gratitude to all the authors of this paper for their contributions. Additionally, we would like to thank the authors of [StegaStamp](https://github.com/tancik/StegaStamp) and [RoSteALS](https://github.com/TuBui/RoSteALS)  for their inspiring work, which played a significant role in the realization of this model.

<hr>

## References
Please <b>CITE</b> our paper whenever this repository is used to help produce published results or incorporated into other software.

@inproceedings{shadmand2024stampone,<br>
    title={StampOne: Addressing Frequency Balance in Printer-proof Steganography},<br>
    author={Shadmand, Farhad and Medvedev, Iurii and Schirmer, Luiz and Marcos, Jo{\~a}o and Gon{\c{c}}alves, Nuno},<br>
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},<br>
    pages={4367--4376},<br>
    year={2024}<br>
}


