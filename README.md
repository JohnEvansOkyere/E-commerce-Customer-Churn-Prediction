# E-Commerce Churn Prediction: A Deep Dive Project
## Objective
The goal of this project is to predict e-commerce customer churn by analyzing key factors such as complaints, satisfaction scores, the number of devices registered, hours spent on the app, warehouse-to-home distance, and city tier. These factors, identified using a heatmap, demonstrated the highest positive correlation with churn.

## Motivation
This project was completed over five days as an opportunity to deepen my understanding of machine learning concepts that I had previously applied without fully grasping their underlying mechanisms. To achieve this, I extensively utilized the following resources:

Scikit-learn Documentation
FreeCodeCamp
DataCamp
GeeksforGeeks
Stack Overflow
ChatGPT
This project served as both a technical challenge and a valuable learning experience.

## Key Challenges and Solutions
1. Preprocessing the Data
Challenges: This was my first time implementing pipelines, which initially felt overwhelming. I decided to use them after learning about their benefits in a Kaggle intermediate course.
Solutions:
Used StandardScaler to scale numerical variables.
Applied SimpleImputer to handle missing values: replacing numerical nulls with the mean and categorical nulls with the mode.
Utilized OneHotEncoder to transform categorical variables into numerical format.
Learning Outcome: Pipelines ensured that preprocessing steps were automated and consistent across the training and testing datasets.
2. Cross-Validation
Challenges: Understanding and implementing cross-validation required a significant learning effort.
Solutions:
Applied 5-fold cross-validation to divide the dataset into five parts. Each fold reserved one part for testing while training on the remaining four, iterating across all folds.
Benefits: Cross-validation reduced overfitting, ensuring the model performed better on unseen data. Overfitting occurs when a model learns the training data too well, leading to poor generalization.
3. Hyperparameter Tuning
Challenges: This was the most difficult part of the project, requiring two days of focused effort to understand and implement.
Solutions:
Used GridSearchCV for hyperparameter tuning due to its systematic approach for structured datasets. GridSearchCV iteratively searched the parameter grid to find the best model configuration.
Learning Outcome: While hyperparameter tuning remains complex, I now understand its role in optimizing model performance and plan to explore it further.
Models and Results
After preprocessing, cross-validation, and hyperparameter tuning, the following models yielded the best results:

Model	Accuracy
XGBoost Classifier	96%
Random Forest Classifier	95%
These results demonstrated the models' strong ability to predict churn based on the identified features.

## Key Learnings
Documentation and Practice: Reading official documentation and experimenting with code were crucial for understanding pipelines, cross-validation, and hyperparameter tuning.
Pipelines: Automating preprocessing steps streamlined the workflow and reduced the chances of data leakage.
Cross-Validation: This technique ensured that the models generalized well to unseen data.
Hyperparameter Tuning: Significantly improved model performance and deepened my appreciation for parameter optimization.
Conclusion
This project has been a mix of fun, challenges, and continuous learning. It allowed me to strengthen my understanding of essential machine learning concepts and techniques. While I am proud of the progress made, I recognize there is still more to learn—especially in hyperparameter tuning and advanced model evaluation.

I plan to continue building projects that challenge me to explore new concepts and refine my skills.

Technologies Used
Python
Scikit-learn
XGBoost
Pandas
NumPy
Matplotlib
Seaborn

Feel free to suggest improvements or raise issues for discussions. 😊
