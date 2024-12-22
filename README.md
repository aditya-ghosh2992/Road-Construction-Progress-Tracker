# SIHTeamBuzz

## Overview
SIHTeamBuzz is a Python-based image processing tool that calculates the progress of road construction by comparing two images taken at different stages of the project. The software uses techniques such as edge detection and structural similarity to estimate the amount of construction progress.

---

## Tech Stack

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-%23F7931E.svg?style=for-the-badge&logo=scikit-image&logoColor=white)
![FPDF](https://img.shields.io/badge/fpdf-FFD700.svg?style=for-the-badge&logo=fpdf&logoColor=black)

---

## Algorithms Used:
1. **Preprocessing with OpenCV:**
   - Images are first resized, converted to grayscale, and normalized to ensure consistency in the analysis.
   
2. **Edge Detection:**
   - The [Canny Edge Detection](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html) algorithm is used to detect edges in road images, helping identify road surface changes and improvements.
   
3. **Structural Similarity Index (SSIM):**
   - The [SSIM](https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html) algorithm is used to compute the similarity between two images. A lower similarity score indicates more significant changes, which suggests construction progress.
   
4. **Road Coverage Change Calculation:**
   - The number of edge pixels in the images is compared to estimate the percentage change in the road surface area, which contributes to understanding the construction progress.

5. **Linear Regression:**
   - Linear regression is used to assist in estimating construction progress based on image analysis results.

---

## Requirements:
- Python 3.x
- OpenCV
- NumPy
- scikit-image
- FPDF

You can install these dependencies using the following commands:

### Install Dependencies:
For all platforms (Mac, Windows, Linux):
```bash
pip install opencv-python numpy scikit-image fpdf
