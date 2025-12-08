<div align="center">

<h3>S<sup>2</sup>M<sup>2</sup>: Scalable Stereo Matching Model for Reliable Depth Estimation (ICCV 2025)</h3>
<h3>Junhong Min¹<sup>*</sup>, Youngpil Jeon¹, Jimin Kim¹, Minyong Choi¹</h3>  
<p>¹Samsung Electronics, <sup>*</sup>Corresponding Author</p>  

<p align="center">
  <a href="https://arxiv.org/abs/2507.13229">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2507.13229-b31b1b?style=for-the-badge&logo=arxiv">
  </a>
  <a href="https://junhong-3dv.github.io/s2m2-project">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-S²M²%20Page-brightgreen?style=for-the-badge&logo=googlechrome">
  </a>
</p>

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="fig/thumbnail.png" width="90%">)

[comment]: <> (</p>)

</div>

---

## 🤗 Notice
>This repository and its contents are **not related to any official Samsung Electronics products**.  
>All resources are provided **solely for non-commercial research and education purposes**.
>
---
## ✨ Key Features
### 🧩 Model
- Scalable stereo matching architecture  
- State-of-the-art performance on ETH3D (1st), Middlebury V3 (1st), and Booster (1st)
- Joint estimation of disparity, occlusion, and confidence
- Supports negative disparity estimation
- Optimal under the pinhole camera model with ideal stereo rectification (vertical disparity < 2px)

### ⚙️ Code
- ✅ FP16 / FP32 inference
- ✅ TorchScript/ONNX/TensorRT export
- ✅ online stereo calibration
- ❌ Training pipeline (not included)

> Note: This implementation replaces the dynamic attention-based refinement module with an UNet for stable ONNX export. It also includes an additional M variant and extended training data with transparent objects. 

---

## 🚀 Performance

> Detailed benchmark results and visualizations are available on the [Project Page](https://junhong-3dv.github.io/s2m2-project).

Inference Speed (FPS)** on **NVIDIA RTX 5090 (float16 + `refine_iter=3`):

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CH</th>
            <th>NTR</th>
            <th>Inference</th>
            <th>640x480</th>
            <th>1216x1024</th>
            <th>2432x2048</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>S</td>
            <td rowspan=2>128</td>
            <td rowspan=2>1</td>
            <td>torch.compile</td>
            <td>59.8</td>
            <td>26.4</td>
            <td>6.6</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>124.0</td>
            <td>59.4</td>
            <td>7.3</td>
        </tr>
        <tr>
            <td rowspan=2>M</td>
            <td rowspan=2>192</td>
            <td rowspan=2>2</td>
            <td>torch.compile</td>
            <td>32.4</td>
            <td>12.5</td>
            <td>3.1</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>66.7</td>
            <td>18.3</td>
            <td>3.8</td>
        </tr>
        <tr>
            <td rowspan=2>L</td>
            <td rowspan=2>256</td>
            <td rowspan=2>3</td>
            <td>torch.compile</td>
            <td>25.8</td>
            <td>7.6</td>
            <td>2.0</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>46.6</td>
            <td>11.2</td>
            <td>2.4</td>
        </tr>
        <tr>
            <td rowspan=2>XL</td>
            <td rowspan=2>384</td>
            <td rowspan=2>3</td>
            <td>torch.compile</td>
            <td>16.3</td>
            <td>4.3</td>
            <td>1.1</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>26.6</td>
            <td>6.4</td>
            <td>1.4</td>
        </tr>
    </tbody>
</table>

---

## 🔧 Installation

We recommend using Python 3.10, PyTorch 2.9, CUDA 12.9, CUDNN 9.1.0, and tensorRT 10.13.3 with Anaconda.

```bash
git clone https://github.com/junhong-3dv/s2m2
cd s2m2

conda env create -n s2m2 -f environment.yml
conda activate s2m2
pip install -e .
```


If the environment setup via .yml doesn’t work smoothly,
you can manually install the main dependencies with:
```bash
pip install torch torchvision opencv-python open3d onnx onnxruntime-gpu onnxscript tensorrt-cu12==10.13.3.9 --extra-index-url https://pypi.nvidia.com/tensorrt-cu12-libs

```
That should cover most of the required packages for running the demo.

## 🚀 Pre-trained Models and Inference

### 1. Download Pre-trained Models

Create a directory for weights and download the desired models from the links below.

```bash
mkdir weights
mkdir weights/pretrain_weights
```

| Model | Download | Model Size |
| :---: | :---: | :--: |
| **S** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth) | 26.5M | 
| **M** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH192NTR2.pth) | 80.4M | 
| **L** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH256NTR3.pth) | 181M | 
| **XL**| [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth) | 406M | 

### 2. Run Basic Demo
To generate a result for a single input, run `demo/visualize_2d_simple.py`.

```bash
python ./demo/visualize_2d_simple.py --model_type XL --num_refine 3 
```
| arg | default | type | help |
| :---: | :---: | :--: | :--: |
| --model_type | 'XL' | str | select model type: [S,M,L,XL] |
| --num_refine | 3 | int | number of local iterative refinement |
| --torch_compile | False | set_true | apply torch_compile  | 
| --allow_negative | False | set_true | allow negative disparity  | 


### 3. Run 3D Visualization Demo

To visualize the 3D output interactively, run `demo/visualize_3d_booster.py` or `demo/visualize_3d_middlebury.py`

```bash
python ./demo/visualize_3d_booster.py --model_type L 
```
For 'visualize_3d_middlebury.py --model_type XL ', result should be like below. 
<p align="center">
  <img src="fig/result_bicycle2.png" width="90%">
</p>
If you failed to reproduce this, let me know. 

## 🚀 Model Optimization (ONNX / TensorRT)

### 1. Export to ONNX (OPSET=18)

Use export_onnx.py to convert the model to ONNX:

```console
python demo/export_onnx.py --model_type $MODEL_TYPE --img_width $IMG_WIDHT --img_height $IMG_HEIGHT
```
### 2. Export to TensorRT (tested with 10.13.3)
Use export_tensorrt.py to build a TensorRT engine:

```console
python demo/export_tensorrt.py --model_type $MODEL_TYPE --img_width $IMG_WIDHT --img_height $IMG_HEIGHT --precision $PRECISION
```
or simply in the terminal
```console
trtexec --onnx={onnx_file_path} --saveEngine=./{trt_file_path} --fp16 --precisionConstraints=obey --layerPrecisions=node_linalg_vector_norm_2:fp32
```
Supported TensorRT precisions: fp32, tf32, fp16

## 📜 Citation

If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{min2025s2m2,
  title={{S\textsuperscript{2}M\textsuperscript{2}}: Scalable Stereo Matching Model for Reliable Depth Estimation},
  author={Junhong Min and Youngpil Jeon and Jimin Kim and Minyong Choi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
