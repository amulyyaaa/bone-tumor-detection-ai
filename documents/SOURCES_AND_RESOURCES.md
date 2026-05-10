# 📚 Sources & Resources — Bone Tumor Detection Project

---

## 🗄️ Dataset Sources

### Primary Recommended Dataset
| | |
|---|---|
| **Name** | Bone Tumor Dataset for Classification |
| **URL** | https://www.kaggle.com/datasets/tushardhumal/bone-tumor-dataset-for-classification |
| **Classes** | tumor / no_tumor |
| **License** | CC0 — Public Domain |

### Alternative Datasets
| Dataset | URL | Notes |
|---------|-----|-------|
| Musculoskeletal X-rays (MURA) | https://stanfordmlgroup.github.io/competitions/mura/ | Stanford, high quality, well-cited |
| Bone Fracture & Tumor Dataset | https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays | Includes fracture classes too |
| RSNA Bone Age | https://www.kaggle.com/c/rsna-bone-age | For bone structure understanding |
| NIH Chest X-Ray | https://www.kaggle.com/datasets/nih-chest-xrays/data | General radiology practice dataset |

---

## 📖 Research Papers & References

### Deep Learning for Medical Imaging
1. **Esteva et al. (2017)** — "Dermatologist-level classification of skin cancer with deep neural networks"  
   *Nature, 542(7639), 115–118*  
   https://www.nature.com/articles/nature21056

2. **Rajpurkar et al. (2018)** — "Deep learning for chest radiograph diagnosis"  
   *PLOS Medicine*  
   https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686

3. **Litjens et al. (2017)** — "A survey on deep learning in medical image analysis"  
   *Medical Image Analysis, 42, 60–88*  
   https://arxiv.org/abs/1702.05747

### MobileNetV2 Architecture
4. **Howard et al. (2017)** — "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"  
   *arXiv:1704.04861*  
   https://arxiv.org/abs/1704.04861

5. **Sandler et al. (2018)** — "MobileNetV2: Inverted Residuals and Linear Bottlenecks"  
   *CVPR 2018*  
   https://arxiv.org/abs/1801.04381

### Explainability (Grad-CAM)
6. **Selvaraju et al. (2017)** — "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"  
   *ICCV 2017*  
   https://arxiv.org/abs/1610.02391

### Transfer Learning
7. **Pan & Yang (2010)** — "A Survey on Transfer Learning"  
   *IEEE Transactions on Knowledge and Data Engineering*  
   https://ieeexplore.ieee.org/document/5288526

### Bone Tumor Classification with AI
8. **Geng et al. (2021)** — "Automated Detection of Bone Tumors Using Deep Learning"  
   https://www.frontiersin.org/articles/10.3389/fonc.2021.603872

9. **Urakawa et al. (2019)** — "Detecting Pathological Fractures Using Deep CNN"  
   *Skeletal Radiology*  
   https://link.springer.com/article/10.1007/s00256-019-3263-3

---

## 🛠️ Technical Documentation

### Core Libraries
| Library | Version | Documentation |
|---------|---------|---------------|
| TensorFlow | 2.12+ | https://www.tensorflow.org/api_docs |
| Keras | (bundled) | https://keras.io/api/ |
| Streamlit | 1.28+ | https://docs.streamlit.io |
| OpenCV | 4.8+ | https://docs.opencv.org |
| scikit-learn | 1.3+ | https://scikit-learn.org/stable/api |
| NumPy | 1.24+ | https://numpy.org/doc |
| Matplotlib | 3.7+ | https://matplotlib.org/stable/api |
| Pillow | 10.0+ | https://pillow.readthedocs.io |

### Pretrained Model
- **MobileNetV2 (ImageNet weights)**: Automatically downloaded by TensorFlow on first run from:  
  https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/

---

## 🎓 Learning Resources

### Free Online Courses
| Course | Platform | URL |
|--------|----------|-----|
| Deep Learning Specialization | Coursera (Andrew Ng) | https://www.coursera.org/specializations/deep-learning |
| TensorFlow Developer Certificate | Coursera | https://www.coursera.org/professional-certificates/tensorflow-in-practice |
| Practical Deep Learning for Coders | fast.ai | https://course.fast.ai |
| CS231n: CNN for Visual Recognition | Stanford | http://cs231n.stanford.edu |

### YouTube Videos (Free)
- **3Blue1Brown — Neural Networks**: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Sentdex — TensorFlow tutorials**: https://www.youtube.com/c/sentdex
- **Streamlit crash course**: https://www.youtube.com/watch?v=_9WiB2PDO7k
- **Transfer Learning explained**: https://www.youtube.com/watch?v=yofjFQddwHE

### Books
- **Deep Learning with Python** — François Chollet (Manning, 2021)
- **Hands-On Machine Learning** — Aurélien Géron (O'Reilly, 3rd ed.)
- **Deep Learning** — Goodfellow, Bengio, Courville (MIT Press, 2016 — free PDF online)

---

## 🔧 Tools Used in This Project

| Tool | Purpose | Download |
|------|---------|----------|
| Python 3.9–3.11 | Primary language | https://python.org |
| VS Code | Code editor | https://code.visualstudio.com |
| Kaggle | Dataset download | https://kaggle.com |
| Google Colab | Alternative GPU training | https://colab.research.google.com |
| GitHub | Version control | https://github.com |
| draw.io | Architecture diagrams | https://app.diagrams.net |

---

## 📊 Where to Submit/Present

- **IEEE Conference on Medical Imaging** (annual)
- **MICCAI** — Medical Image Computing and Computer Assisted Intervention
- **arXiv.org** — Preprint server (free to submit)
- College project expo / department showcase

---

## 🔗 Useful GitHub Repositories (for reference)

- https://github.com/keras-team/keras-applications (Keras pretrained models)
- https://github.com/jacobgil/pytorch-grad-cam (Grad-CAM implementation reference)
- https://github.com/streamlit/streamlit (Streamlit source)
- https://github.com/topics/bone-tumor-detection (similar projects)

---

*All dataset licenses should be verified before commercial use. This project is for educational and research purposes.*
