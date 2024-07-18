# K Nearest Neighbors (KNN) Project

This project demonstrates the implementation of the K Nearest Neighbors (KNN) algorithm to predict the target class using anonymous data features. The project includes data preprocessing, exploratory data analysis, model training, and evaluation. The primary goal is to determine the optimal number of neighbors (k) using the elbow method and to evaluate the model's performance using classification metrics.

## Project Structure

- `K Nearest Neighbors Project.ipynb`: The Jupyter Notebook containing the entire code for data preprocessing, model training, evaluation, and visualization.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- **Pairplot**: Created a pairplot to visualize the relationships between the features and to understand the data distribution. Due to the anonymity of the data, the exact column names and their meanings are unknown.

### 2. Data Preprocessing
- **Standardization**: Standardized the features to ensure that all variables contribute equally to the distance calculations in the KNN algorithm.

### 3. Model Training and Evaluation
- **Train-Test Split**: Split the data into training and testing sets to evaluate the model's performance on unseen data.
- **KNN Instantiation and Predictions**: 
  - Instantiated a KNN model with `neighbors=1` and computed the predictions.
  - Repeated the process for k values ranging from 1 to 40.
- **Elbow Method**: Plotted the graph of K-values vs. error rates to determine the optimal k value.
- **Retraining with Optimal k**: 
  - Retrained the model using the optimal k value obtained from the elbow method.
  - Fitted the model and computed the predictions.
- **Model Evaluation**: Evaluated the model using a classification report and confusion matrix to assess its performance.

## Usage

1. **Clone the Repository**: Clone this repository to your local machine.
   ```bash
   git clone https://github.com/anishapareek/k-nearest-neighbors.git
   ```

2. **Install Dependencies**: Ensure you have the necessary libraries installed. You can install them using pip.
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

3. **Run the Jupyter Notebook**: Open the Jupyter Notebook (`K Nearest Neighbors Project.ipynb`) and run the cells to see the implementation and results.

## Key Findings

- **Optimal k Value**: The elbow method helped in determining the optimal k value that balances bias and variance, providing the best performance for the KNN model.
- **Model Performance**: The classification report and confusion matrix provided insights into the model's accuracy, precision, recall, and F1-score, indicating its effectiveness in predicting the target class.

## Conclusion

This project showcases the step-by-step implementation of the KNN algorithm, emphasizing the importance of choosing the right k value and standardizing the features for better model performance. The use of the elbow method for hyperparameter tuning and thorough evaluation using classification metrics ensures a robust and reliable predictive model.
