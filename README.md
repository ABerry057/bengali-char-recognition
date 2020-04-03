
# Bengali Character Regcognition

![Sample Bengali Characters](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/consonant.png)

## Objective
The goal of this project was to use deep learning methods to develop a convolutional neural network architecture capable of correctly classifying characters of the Bengali alphabet. In addition to improving the deep learning skills of the authors, this project will hopefully accelerate Bengali handwritten optical character recognition research and help enable the digitalization of educational resources.

### Methods Used
* Exploratory Data Analysis
* Image Preprocessing
* Convolutional Neural Networks
* Hyperparameter Tuning

### Required Technologies and Packages
* Python 3.6.7
* cv2 4.2.0
* json 2.0.9
* keras 2.2.4
* numpy 1.18.1
* pandas 1.0.1
* PIL 7.0.0
* pyarrow 0.14.1
* seaborn 0.10.0
* tensorflow : 2.2.0
* tqdm 4.38.0

## Project Description
This project involves classifying handwritten characters of the Bengali alphabet, similar to classifying integers in the MNIST data set. In particular, the Bengali alphabet is broken down into three components for each grapheme, or character: 1) the root, 2) the vowel diacritic, 3) the consonant diacritic, where a diacritic is similar to an accent. The goal is the create a classification model that can classify each of these three components of a handwritten grapheme, and the final result is measured using the recall metrics, with double weight given to classification of the root. Click [here](https://www.kaggle.com/c/bengaliai-cv19/overview) for more information.

Please also refer to [our blog](https://aberry057.github.io/) for more information about the authors' process for developing this model.


## Repository Organization and Getting Started

```
├── data/
├── figures/
├── models/
├── reports/
├── src/
├── LICENSE
└── README.md
```

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Find the competition background and data [here](https://www.kaggle.com/c/bengaliai-cv19/overview).
3. All figures, including EDA, are found [here](https://github.com/csjasonchan357/bengali-char-recognition/tree/master/figures))
4. Classification models are found [here](https://github.com/csjasonchan357/bengali-char-recognition/tree/master/models))
5. Blog posts in notebook format are found [here](https://github.com/csjasonchan357/bengali-char-recognition/tree/master/reports))
6. See the LICENSE for permitted use or redistribution of this software

## Contributors

Primary Authors

* Alexander Berry
* Jason Chan
* Hyunjoon Lee

Code contributions from
* [CodeNinja](https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn): inspiration for exploratory data analysis approaches and baseline model architecture



