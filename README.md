# Brain Tumor MRI Analysis and AI Application

## Project Overview
This project focuses on the analysis of brain tumor MRI images and the development of an AI application that can detect and provide information about brain tumors from MR images.

## Dataset
The dataset used in this project is the "Brain Tumor MRI Dataset" from Kaggle, created by Masoud Nickparvar. It contains MRI scans of brains categorized into four classes:
- Glioma
- Meningioma
- No tumor
- Pituitary

Dataset Link: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Details
- The dataset contains 7023 images of human brain MRI scans
- Images are classified into 4 classes: glioma, meningioma, no tumor, and pituitary
- The dataset is divided into training and testing sets

## Project Structure
- `data/`: Contains the MRI dataset
- `notebooks/`: Jupyter notebooks for data analysis and model development
  - `01_data_exploration.ipynb`: Initial data exploration and visualization
  - `02_model_development.ipynb`: Development of the AI model
- `src/`: Source code for the AI application
- `README.md`: Project documentation

## Project Goals
1. Perform comprehensive data analysis on the brain tumor MRI dataset
2. Develop a machine learning model to classify brain tumors
3. Create an application that can analyze MR images and provide diagnostic information

## Getting Started
1. Clone this repository
2. Download the dataset from Kaggle and place it in the `data/` directory
3. Install the required dependencies
4. Run the Jupyter notebooks for analysis and model development

## Dependencies
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by Masoud Nickparvar on Kaggle
