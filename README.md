# Otto-Group-Product-Classification-Challenge

Creating Stacking Classifier comprised of few base learners.

### Table of Contents

#### 1. **Information**
    - Reason for Choosing this Dataset ?
    - Source
    - Details
    - Questionnaire
    - Objective

#### 2. **Loading Dataset**
    - Importing packages
    - Reading Data
    - Shape of data
    - Examining Null values

#### 3. **Data Preparation & EDA**
    - Descriptive Statistics
    - Correlation Heatmap
    - Target Class count
    - Target encoding
    - Standardization

#### 5. **Stacking**
    - Splitting Data & Choosing Algorithms
    - Naive Bayes Classifier
    - Decision Tree Classifier
    - SGD Classifier
    - KNN Classifier
    - MLP Classifier
    - Prediction on Validation and test set
    - Meta-Model as Random Forest Classifier
    - Hyper-parameter tuning Random Forest
    - Prediction on Final Test Set

#### 6. **Conclusion**

#### 7. **What's next ?**<br><br>


### Reason for Choosing this Dataset ?

- The Reason behind choosing this model is my Personal Interest to explore various Domains out there.


- However, this Statistical models are not prepared to use for production environment.


### Source :

- https://www.kaggle.com/c/otto-group-product-classification-challenge/data


### Details :

- The Otto Group is one of the world’s biggest e-commerce companies, with subsidiaries in more than 20 countries, including Crate & Barrel (USA), Otto.de (Germany) and 3 Suisses (France). We are selling millions of products worldwide every day, with several thousand products being added to our product line.


- A consistent analysis of the performance of our products is crucial. However, due to our diverse global infrastructure, many identical products get classified differently. Therefore, the quality of our product analysis depends heavily on the ability to accurately cluster similar products. The better the classification, the more insights we can generate about our product range.


- ![alt text](https://storage.googleapis.com/kaggle-competitions/kaggle/4280/media/Grafik.jpg)


- For this competition, we have provided a dataset with 93 features for more than 200,000 products. The objective is to build a predictive model which is able to distinguish between our main product categories. The winning models will be open sourced.


- Evaluation Metrics :

    - Submissions are evaluated using the multi-class logarithmic loss. Each product has been labeled with one true category. For each product, you must submit a set of predicted probabilities (one for every category). The formula is then,
$$log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}),$$
    - where N is the number of products in the test set, M is the number of class labels, \\(log\\) is the natural logarithm, \\(y_{ij}\\) is 1 if observation \\(i\\) is in class \\(j\\) and 0 otherwise, and \\(p_{ij}\\) is the predicted probability that observation \\(i\\) belongs to class \\(j\\).
    - The submitted probabilities for a given product are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).


- Submission Format :
    - You must submit a csv file with the product id, all candidate class names, and a probability for each class. The order of the rows does not matter. The file must have a header and should look like the following:
    - id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9<br>
      1,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0<br>
      2,0.0,0.2,0.3,0.3,0.0,0.0,0.1,0.1,0.0,.etc.<br>
      
      
### Questionnaire :

- How is our Target variable distributed ? is it Imbalanced ?

### Objective :

- The goal is to make Stacked predictive model, understanding the intuition behinf it and reviewing some exploratory and modelling techniques.
