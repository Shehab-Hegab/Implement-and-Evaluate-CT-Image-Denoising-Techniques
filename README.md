
# 🧼 CT Image Denoising Techniques – Implementation & Evaluation

This project presents the implementation and performance evaluation of several denoising techniques applied to Computed Tomography (CT) images. The aim is to reduce image noise while preserving clinically significant anatomical details, which is crucial in medical imaging applications.

---

## 📂 Project Structure

```bash
Implement-and-Evaluate-CT-Image-Denoising-Techniques/
│
├── data/                   # Datasets and loading utilities
├── denoising_methods/      # Implemented filters (Gaussian, Median, etc.)
├── evaluation/             # Metrics (PSNR, SSIM, MSE)
├── results/                # Denoised images and performance charts
├── utils/                  # Helper functions and utilities
├── main.py                 # Main runner script
└── requirements.txt        # Python dependencies
````

---

## 🧪 Implemented Denoising Techniques

* ✅ Gaussian Filter
* ✅ Median Filter
* ✅ Bilateral Filter
* ✅ Non-Local Means (NLM)
* ✅ Wavelet Denoising
* 🔄 *Future*: Deep Learning methods (e.g., DnCNN, UNet)

---

## 📊 Evaluation Metrics

The following metrics were used to assess the effectiveness of each denoising method:

* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index)
* **MSE** (Mean Squared Error)

---

## 📷 Visual Results

Below are sample comparisons between noisy CT images and results after applying various denoising techniques:

<p align="center">
  <img src="https://github.com/user-attachments/assets/784e1d36-e200-4430-9391-03236cdcf9df" width="450">
  <img src="https://github.com/user-attachments/assets/21a893e8-63b0-4a43-8ad3-e6ab43e26a25" width="450">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9be166d5-7cd1-4d44-98c3-b5f3898dbcbd" width="450">
  <img src="https://github.com/user-attachments/assets/4baaa85b-2cb3-4bff-9999-ee489749ba98" width="450">
</p>

More results are available in the `results/` folder.

---

## 📥 Dataset Sources

We used publicly available CT image datasets suitable for image denoising and diagnostic imaging research:

* 🧪 [Kaggle: CT Scans (COVID)](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans/data)
* 🏥 [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
* 📁 Custom images (from hospital archives or academic sources) placed in `data/`

> Ensure the dataset structure follows the guide in `data/README_data.txt`.

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/Shehab-Hegab/Implement-and-Evaluate-CT-Image-Denoising-Techniques.git
   cd Implement-and-Evaluate-CT-Image-Denoising-Techniques
   ```

2. **Install the required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project pipeline**

   ```bash
   python main.py
   ```

4. **Check results in the `/results` folder**

---

## 🛠 Dependencies

* Python 3.8+
* NumPy
* OpenCV
* scikit-image
* matplotlib
* tqdm

Install via:

```bash
pip install -r requirements.txt
```

---

![WhatsApp Image 2025-05-03 at 18 51 23_fc25f49c](https://github.com/user-attachments/assets/716455f0-7d69-450f-8e67-d24a21ebcae5)




![image](https://github.com/user-attachments/assets/ff7f2dc5-8ea7-45fe-9122-6270326e33f0)
## 📈 Quantitative Results

| Denoising Method | PSNR ↑ | SSIM ↑ | MSE ↓ |
| ---------------- | ------ | ------ | ----- |
| Gaussian Filter  | 28.9   | 0.75   | 145.3 |
| Median Filter    | 30.1   | 0.79   | 128.2 |
| Bilateral Filter | 31.2   | 0.82   | 114.5 |
| Non-Local Means  | 32.5   | 0.85   | 99.7  |

---

## 📌 Future Enhancements

* 📦 Deep learning integration (DnCNN, U-Net)
* 🧠 Adaptive parameter tuning based on noise levels
* 🔀 Real-time denoising for clinical use
* 🧬 Evaluation on multimodal data (MRI, PET, SPECT)

---

## 👨‍💻 Author

**Shehab Hegab**
Biomedical Engineer & Machine Learning Engineer
📍 Cairo, Egypt
🔗 [LinkedIn](https://www.linkedin.com/in/shehab-hegab/)
🌐 [GitHub](https://github.com/Shehab-Hegab)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.



