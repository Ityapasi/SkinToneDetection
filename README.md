# 🎨 Real-Time Skin-Tone Region Detection & Color Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Processing-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A real-time Computer Vision system that isolates skin regions using a hybrid dual-color-space pipeline (**HSV + YCrCb**), performs morphological noise cleanup, extracts precise color metrics, and dynamically generates tone classifications with complementary styling swatches.

---

## 📌 Features

- **Dual Color-Space Segmentation:** Combines **HSV** (illumination-decoupled chrominance) and **YCrCb** (chroma-red / chroma-blue isolation) for high-accuracy skin extraction under varied lighting conditions.
- **Morphological Artifact Removal:** Dual-pass morphological opening (`cv2.MORPH_OPEN`) and closing (`cv2.MORPH_CLOSE`) to eliminate background clutter and fill specular reflections (glasses glare, forehead highlights).
- **Temporal Color Smoothing (EMA):** Uses an Exponential Moving Average algorithm ($\alpha = 0.15$) to prevent frame-to-frame RGB/HEX flicker caused by auto-exposure.
- **Live Color Swatch UI:** Dynamically displays a live color preview card, extracted HEX code, RGB breakdown, and detected skin-tone category.
- **Styling Palette Generator:** Automatically computes and recommends 3 complementary accent colors based on detected luminance.
- **Real-Time Performance:** Includes a live FPS counter processing at 30+ frames per second on standard CPU threads.

---

## 🛠️ Pipeline Architecture

```text
 Webcam Frame
      │
      ▼
 Gaussian Blur (Noise Reduction)
      │
 ┌────┴──────────────────────────┐
 │                               │
 ▼                               ▼
HSV Thresholding          YCrCb Thresholding
[H: 0-25, S: 30-255]     [Cr: 133-173, Cb: 77-127]
 │                               │
 └──────────────┬────────────────┘
                ▼
         Bitwise AND Mask
                │
                ▼
 Morphological Filtering (Open & Close)
                │
                ▼
    Contour Area Filtering (> 3500px)
                │
                ▼
 Color Extraction & EMA Smoothing
                │
                ▼
 Tone Classification & UI Card Overlay