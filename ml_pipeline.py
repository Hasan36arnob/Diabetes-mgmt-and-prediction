# Import necessary libraries
# import pandas as pd  # For data manipulation and analysis
# # from sklearn.model_selection import train_test_split  # To split dataset into training and testing sets
# from sklearn.preprocessing import LabelEncoder  # To encode categorical variables into numeric
# from sklearn.metrics import classification_report, accuracy_score  # To evaluate model performance
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier  # Ensemble models
# from sklearn.linear_model import LogisticRegression, RidgeClassifier  # Linear models
# from sklearn.naive_bayes import GaussianNB  # Naive Bayes classifier
# from sklearn.neural_network import MLPClassifier  # Multi-layer perceptron (neural network)
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA model
# import matplotlib.pyplot as plt  # For visualizing results

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt





# # Step 1: Load the dataset
# data = pd.read_csv('diabetess.csv')  # Load the dataset from a CSV file
data = pd.read_csv('diabetess.csv') # Load the dataset from a CSV file

# Step 2: Preprocess the dataset
# label_encoder = LabelEncoder()  # Initialize the LabelEncoder for encoding categorical variables
# data_encoded = data.copy()  # Create a copy of the dataset to avoid modifying the original data

# # Loop through each column and encode if it is categorical (dtype == 'object')
# for col in data_encoded.columns:
#     if data_encoded[col].dtype == 'object':  # Check if the column is categorical
#         data_encoded[col] = label_encoder.fit_transform(data_encoded[col])  # Encode the column
        
label_encoder = LabelEncoder()
data_encoded = data.copy()

for col in data_encoded.columns:
    if data_encoded[col].dtype == 'object':
        data_encoded[col] = label_encoder.fit_transform(data_encoded[col])       
         
        

# # Step 3: Verify the column names
# print("Columns in the dataset:", data_encoded.columns)  # Print the columns to identify the target variable



# Step 4: Assign the target column
# target_column = 'class'  # Replace 'class' with the actual target column name if it differs in your dataset

target_column = 'class'

# # Step 5: Split the dataset into features and target
# X = data_encoded.drop(columns=[target_column])  # Features (all columns except the target column)
# y = data_encoded[target_column]  # Target (the column specified as the target)

X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column] 

# Step 6: Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )  # 80% training and 20% testing, with stratification to maintain class balance

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state = 42 
    
)


# # Step 7: Define machine learning models
# models = {
#     "Random Forest": RandomForestClassifier(random_state=42),  # Random Forest Classifier
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),  # Gradient Boosting
#     "AdaBoost": AdaBoostClassifier(random_state=42),  # AdaBoost
#     "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),  # Logistic Regression
#     "Ridge Classifier": RidgeClassifier(random_state=42),  # Ridge Classifier
#     "Naive Bayes": GaussianNB(),  # Naive Bayes
#     "MLP Classifier": MLPClassifier(max_iter=1000, random_state=42),  # Neural Network
#     "LDA": LinearDiscriminantAnalysis()  # Linear Discriminant Analysis
# }

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
     
    "AdaBoost" : AdaBoostClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "MLP Classifier": MLPClassifier(max_iter=1000, random_state=42),
    "LDA": LinearDiscriminantAnalysis() 
    
}


# Step 8: Train and evaluate each model
# results = {}  # Dictionary to store results
# for name, model in models.items():  # Iterate over each model
#     model.fit(X_train, y_train)  # Train the model
#     predictions = model.predict(X_test)  # Make predictions on the test set
#     report = classification_report(y_test, predictions, output_dict=True)  # Generate classification report
#     accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy score
#     # Store metrics (accuracy, precision, recall, f1-score) for the model
#     results[name] = {
#         "accuracy": accuracy,
#         "precision": report['weighted avg']['precision'],
#         "recall": report['weighted avg']['recall'],
#         "f1-score": report['weighted avg']['f1-score']
#     }
    
results = {}
for name , model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict = True)
    accuracy = accuracy_score(y_test, predictions)
    
    results[name] = {
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score']
    }  
    

# # Step 9: Convert results to a DataFrame and sort by accuracy
# results_df = pd.DataFrame(results).T  # Convert results dictionary to DataFrame
# results_df = results_df.sort_values(by="accuracy", ascending=False)  # Sort by accuracy in descending order

results_df = pd.DataFrame(results).T 
results_df = results_df.sort_values(by="accuracy", ascending=False)

# Step 10: Print model performance comparison
# print("Model Performance Comparison:")
# print(results_df)  # Display the performance metrics for each model

print("Model Performance Comparison:")
print(results_df) 

# Step 11: Visualize the results
# plt.figure(figsize=(10, 6))  # Set the figure size
# plt.bar(results_df.index, results_df['accuracy'], color='skyblue')  # Create a bar plot for accuracy
# plt.title('Model Comparison Based on Accuracy')  # Title of the plot
# plt.ylabel('Accuracy')  # Label for y-axis
# plt.xlabel('Models')  # Label for x-axis
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
# plt.show()  # Display the plot

 # Plot results with better visualization
plt.figure(figsize=(12, 6))  # Adjust the figure size for better readability
plt.bar(results_df.index, results_df['accuracy'], color='red')  # Bar plot

# Set title and labels
plt.title('Model Comparison', fontsize=16)  # Larger font size for the title
plt.ylabel('Accuracy', fontsize=14)  # Larger font size for the y-axis label
plt.xlabel('Models', fontsize=14)  # Larger font size for the x-axis label

# Adjust x-axis labels
plt.xticks(rotation=30, ha='right', fontsize=12)  # Rotate labels and align to the right

# Display the plot
plt.tight_layout()  # Adjust layout to avoid clipping
plt.show()
