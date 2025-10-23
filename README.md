<div align="center">

<h1>S<sup>2</sup>M<sup>2</sup>: Scalable Stereo Matching Model for Reliable Depth Estimation</h1>

<h3>Junhong Min¹<sup>*</sup>, Youngpil Jeon¹, Jimin Kim¹, Minyong Choi¹</h3>  
<p>¹Samsung Electronics, <sup>*</sup>Corresponding Author</p>  
<p><strong>ICCV 2025</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.13229">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2507.13229-b31b1b?style=for-the-badge&logo=arxiv">
  </a>
  <a href="https://junhong-3dv.github.io/s2m2-project">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-S²M²%20Page-brightgreen?style=for-the-badge&logo=googlechrome">
  </a>
</p>

<p align="center">
  <img src="fig/thumbnail.png" width="90%">
</p>

</div>

---

## 🤗 Notice

The demo code and pre-trained models will be released by **end of October 2025**. Stay tuned!

---

## ✨ Key Features

### 🧩 Model

- Scalable stereo matching architecture  
- State-of-the-art performance on ETH3D (1st), Middlebury V3 (1st), and Booster (1st)
- Joint estimation of disparity, occlusion, and confidence
- Supports negative disparity estimation

### ⚙️ Code

- ✅ FP16 / FP32 inference  
- ✅ ONNX export  
- ✅ TorchScript export  
- ❌ Training pipeline (not included)

> Note: This implementation replaces the dynamic attention-based refinement module with a lightweight UNet for stable ONNX export. It also includes an additional M variant and extended training data with transparent objects.

---

## 🏛️ Model Architecture

<p align="center">
  <img src="fig/overview.png" width="90%">
</p>

---

## 🚀 Performance

> Detailed benchmark results and visualizations are available on the [Project Page](https://junhong-3dv.github.io/s2m2-project).

Inference Speed (FPS)** on **NVIDIA RTX 4090 (float16 + `torch.compile` + `refine_iter=3`):

| Model | CH  | NTR | 640×480 | 1216×1024 | 2432×2048 |
|-------|-----|-----|---------|-----------|-----------|
| S     | 128 | 1   | 68.7    | 18.8      | 3.9       |
| M     | 192 | 2   | 34.8    | 8.9       | 1.9       |
| L     | 256 | 3   | 20.5    | 5.3       | 1.2       |
| XL    | 384 | 3   | 11.2    | 2.7       | 0.64      |

---

## 🔧 Installation

We recommend using Python 3.10 and PyTorch 2.4+ with Anaconda.

```bash
git clone https://github.com/junhong-3dv/s2m2.git
cd s2m2

conda env create -f environment.yml
conda activate s2m2
