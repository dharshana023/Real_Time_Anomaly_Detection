Real-Time Anomaly Detection in Network Security
**Installation Procedure**
Prerequisites
To execute this project, ensure you have the following installed on your system:

Python 3.x
Flask: For building the web interface.
pandas: For data manipulation.
scikit-learn: For machine learning model training and evaluation.
matplotlib & seaborn: For visualizations.
UNSW-NB15 Dataset: For anomaly detection training.

** Dataset Information**
Dataset: UNSW-NB15
Features: 49
Classes: 9 (Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Worms)
Size: 2 million data points
Provided by: Australian Centre for Cyber Security
Type: Modern network traffic dataset with both normal and attack records.
The dataset contains a wide range of attack patterns and malicious activities, including Denial of Service (DoS), Reconnaissance, and others, which provides a comprehensive basis for training the anomaly detection system


**49 features with various types of attacks such as:**

Fuzzers
Analysis
Backdoors
DoS
Exploits
Generic
Reconnaissance
Shellcode
Worms
**MODEL EXPLAINATION:**
**1. RandomForestClassifier**
Description:
The RandomForestClassifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification. It is designed to improve the stability and accuracy of decision trees by reducing overfitting and handling imbalanced data better. The randomness in selecting features and data subsets ensures robustness and generalizability to new data.

Why Chosen:
RandomForest was selected due to its ability to handle large datasets with high-dimensional features. It also excels in managing imbalanced datasets, which is crucial for anomaly detection where normal data significantly outnumbers the anomalies.

Model Performance:

Accuracy: 90.54%
Precision: 90.43%
Recall: 90.54%
F1 Score: 90.48%
Tuning Strategies:

Number of Estimators: 100 trees were used, balancing between computational cost and performance.
Max Depth: Increased depth helped improve recall for minority classes, especially in identifying anomalies. However, tuning depth carefully prevents overfitting.
Max Features: 'sqrt' (square root of total features) was selected for each split, ensuring a balance between variance reduction and speed.
Inference:
RandomForest delivered high overall accuracy and performed well in detecting anomalies due to its ensemble nature, which reduces bias and variance. However, some minority classes (e.g., Backdoor and Worms) still had low recall, indicating that further tuning or feature engineering could improve detection for rare attacks.

**2. DecisionTreeClassifier**
Description:
A Decision Tree is a non-parametric model that splits the data based on feature conditions, leading to terminal nodes that represent the predicted output. It is interpretable, fast, and effective for small datasets but can be prone to overfitting, especially in complex tasks like anomaly detection.

Why Chosen:
Decision Trees are easy to understand and interpret, making them ideal for initial explorations. Although they may not handle complex patterns as well as ensemble methods, their speed in training and low computational cost make them suitable for quick prototyping.

Model Performance:

Accuracy: 90.21%
Precision: 90.21%
Recall: 90.21%
F1 Score: 90.21%
Tuning Strategies:

Max Depth: Restricted to prevent overfitting and control the complexity of the tree.
Min Samples Split: Set to a higher value (10) to ensure splits only happen when sufficient samples are available, preventing overfitting on noisy data.
Inference:
While the Decision Tree model achieved a good level of accuracy, it performed worse than RandomForest in detecting complex patterns due to its single-tree structure. It struggled with rare anomalies like Backdoor and Worms, which highlights the need for ensemble methods in these cases.

**3. KNeighborsClassifier**
Description:
The KNeighborsClassifier is a distance-based algorithm that classifies data points based on their proximity to other labeled data points. The number of neighbors (k) defines how many surrounding points are considered when making a prediction.

Why Chosen:
KNN is intuitive for anomaly detection since anomalies are often far from normal data points in feature space. However, it can be computationally expensive as the dataset grows, and performance degrades with increasing dimensionality.

Model Performance:

Accuracy: 86.05%
Precision: 81.57%
Recall: 86.05%
F1 Score: 83.50%
Tuning Strategies:

Number of Neighbors: Set to 5 based on standard practice, but experimented with values from 3 to 15. Higher values of k provided smoother decision boundaries but decreased sensitivity to anomalies.
Distance Metrics: Euclidean distance was used, but alternate metrics like Manhattan could have been tested for further tuning.
Inference:
KNN had the lowest performance compared to the other models, as it is sensitive to noise and computationally expensive with larger datasets. It performed well on more frequent classes but struggled with rare classes, suggesting it is less suited for high-dimensional and imbalanced anomaly detection tasks.

**4. AdaBoostClassifier**
Description:
AdaBoost is a boosting algorithm that combines multiple weak learners (usually decision trees) to form a strong classifier. Each subsequent model attempts to correct the errors of its predecessor, thus improving performance over iterations.

Why Chosen:
AdaBoost was selected for its ability to increase accuracy and reduce bias by focusing on difficult-to-classify samples. It’s well-suited for structured datasets with many categorical features, as it adapts to improve weak base learners.

Model Performance:

Accuracy: 90.95%
Precision: 87.23%
Recall: 90.95%
F1 Score: 88.27%
Tuning Strategies:

Number of Estimators: Set to 100 to balance between accuracy and computational cost.
Base Estimator: DecisionTreeClassifier with max depth of 1 (stumps) was used as the weak learner. Shallow trees help prevent overfitting while allowing AdaBoost to focus on difficult examples.
Inference:
AdaBoost performed well, with accuracy close to RandomForest. However, it struggled with the recall for rare classes, indicating that while it improved precision, it might miss certain anomalies.

**5. XGBClassifier**
Description:
XGBoost is an optimized implementation of the gradient boosting algorithm. It is known for its speed, performance, and ability to handle missing data and large-scale datasets effectively. It combines many weak learners to minimize errors iteratively.

Why Chosen:
XGBoost was selected for its state-of-the-art performance in classification tasks, especially in dealing with high-dimensional data and complex patterns. It offers flexibility and high efficiency for real-time systems. Additionally, XGBoost is chosen because it reduces false positive values, increases overall accuracy, and is well-suited for large-scale and real-time data.

Model Performance:

Accuracy: 93.27%
Precision: 91.41%
Recall: 93.27%
F1 Score: 91.57%
Tuning Strategies:

Learning Rate: Set to 0.1 for optimal convergence speed.
Max Depth: Set to 6, as deeper trees may overfit.
Regularization: L2 regularization (Ridge) was applied to reduce overfitting.
Subsampling: Set to 0.8 to avoid overfitting by training on a random subset of data at each iteration.
Inference:
XGBoost achieved the best performance overall, with high accuracy, precision, recall, and F1 score. It successfully handled the complex patterns in the dataset, especially in detecting rare classes. This makes it the top choice for real-time anomaly detection, particularly due to its ability to reduce false positives while maintaining high accuracy, making it suitable for large and real-time data environments.



