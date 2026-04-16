# Person Attribute Recognition (PAR)

This project implements a multi-task deep learning pipeline for Person Attribute Recognition (PAR), where a single model predicts multiple human attributes from cropped person images.

## 🚀 Attributes Predicted
- Age
- Headgear
- Gender
- Glasses
- Upper-body clothing color
- Lower-body clothing color

## 🧠 Approach

Instead of training separate models for each attribute, this project uses a **multi-task learning approach**, where a shared backbone extracts features and multiple classification heads predict different attributes simultaneously.

This improves:
- Efficiency (single model instead of many)
- Feature sharing across tasks
- Generalization performance

## ⚙️ Pipeline

1. Data loading and preprocessing  
2. Label encoding  
3. Train / Validation / Test split  
4. Multi-task model training  
5. Per-attribute evaluation (Accuracy, F1, Recall)  
6. Test prediction CSV generation  
7. External image inference + annotation  

## 📊 Results

| Attribute | Accuracy | F1 Score |
|----------|--------|---------|
| Age | — | — |
| Gender | — | — |
| Headgear | — | — |
| Glasses | — | — |
| Upper Clothing | — | — |
| Lower Clothing | — | — |

*(Fill this with your actual results)*

## 🖼️ Example Output

(Add prediction images here if possible)

## 🛠️ Tech Stack
- Python
- PyTorch
- OpenCV
- Pandas / NumPy

## 📌 Use Cases
- Surveillance systems
- Retail analytics
- Smart city monitoring
- Pedestrian analysis

## 📈 Future Improvements
- Use transformer backbone (ViT / Swin)
- Handle class imbalance better
- Add attention mechanisms
- Deploy as API (FastAPI)

## 👨‍💻 Author
Rajasekar R
