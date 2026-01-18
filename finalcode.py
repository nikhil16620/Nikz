import pandas as pd

df = pd.read_csv("/content/dataset.csv")

print(df.head())
print(df.info())
print(df.nunique())
     
     # Target column
target_col = "Fraud_Label"  # 1 means fraud, 0 means legit transaction
id_cols = ["Transaction_ID"]






for c in id_cols:
    if c in df.columns:
        df = df.drop(columns=[c])




#  split everything into X (features) and y (what we want to predict)
X = df.drop(target_col, axis=1)  # Everything except the fraud label
y = df[target_col]



print("X shape:", X.shape)
print("y value counts:\n", y.value_counts(normalize=True))  # Checking class balance



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd




# 1. Split data into train and test sets
# Using stratify to keep the same fraud/non-fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 80/20 split seems standard
    stratify=y,
    random_state=42  # For reproducibility
)




print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train fraud ratio:", y_train.mean())
print("Test fraud ratio:", y_test.mean())




# --- Cleaning up the data before we scale ---




if 'User_ID' in X_train.columns:
    X_train = X_train.drop(columns=['User_ID'])
    X_test = X_test.drop(columns=['User_ID'])




if 'Timestamp' in X_train.columns:
    # Convert to datetime format first
    X_train['Timestamp'] = pd.to_datetime(X_train['Timestamp'])
    X_test['Timestamp'] = pd.to_datetime(X_test['Timestamp'])




    # Extract features t - fraud patterns could vary by time
    X_train['hour'] = X_train['Timestamp'].dt.hour
    X_train['dayofweek'] = X_train['Timestamp'].dt.dayofweek
    X_train['month'] = X_train['Timestamp'].dt.month
    X_train['dayofyear'] = X_train['Timestamp'].dt.dayofyear




    X_test['hour'] = X_test['Timestamp'].dt.hour
    X_test['dayofweek'] = X_test['Timestamp'].dt.dayofweek
    X_test['month'] = X_test['Timestamp'].dt.month
    X_test['dayofyear'] = X_test['Timestamp'].dt.dayofyear




    #  drop the original timestamp column
    X_train = X_train.drop(columns=['Timestamp'])
    X_test = X_test.drop(columns=['Timestamp'])




# Finding categorical columns
categorical_cols = X_train.select_dtypes(include='object').columns




# Convert categorical variables to numbers using one-hot encoding
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)




#  train and test have the same columns after encoding
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)




# Add missing columns to test set with zeros
missing_in_test = list(train_cols - test_cols)
for c in missing_in_test:
    X_test[c] = 0




# Add missing columns to train set with zeros
missing_in_train = list(test_cols - train_cols)
for c in missing_in_train:
    X_train[c] = 0




X_test = X_test[X_train.columns]




# --- Done with preprocessing ---




# 2. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score




# 1. Setting up our models - starting with baseline before trying SMOTE
# Logistic Regression - simple but effective
lr = LogisticRegression(
    max_iter=1000,  # Sometimes needs more iterations to converge
    class_weight="balanced",
    random_state=42
)




# Random Forest - usually performs better but takes longer to train
rf = RandomForestClassifier(
    n_estimators=300,  # More trees = better performance usually
    max_depth=None,
    class_weight="balanced",   # Again, dealing with class imbalance
    random_state=42,
    n_jobs=-1
)




# 2. Train both models on our scaled data
print("Training Logistic Regression...")
lr.fit(X_train_scaled, y_train)

print("Training Random Forest...")
rf.fit(X_train_scaled, y_train)




# 3. Function to evaluate model performance

def evaluate_model(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]  # Get probabilities for ROC-AUC

    print(f"\n=== {name} ===")
    print("Confusion matrix:\n", confusion_matrix(y_t, y_pred))
    print("Classification report:\n", classification_report(y_t, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_t, y_proba))  # ROC-AUC is important for imbalanced datasets




# 4.  models perfomance on the test set
evaluate_model("Logistic Regression (baseline)", lr, X_test_scaled, y_test)
evaluate_model("Random Forest (baseline)", rf, X_test_scaled, y_test)
     

from imblearn.over_sampling import SMOTE
import numpy as np

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

# check to see class distribution before/after
print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_sm))

# Setting up Logistic Regression
# Removed class_weight since SMOTE already handled the imbalance
lr_sm = LogisticRegression(
    max_iter=1000,
    class_weight=None,  # not needed anymore
    random_state=42
)

# Random Forest configuration
# Bumped n_estimators to 300 for better performance
rf_sm = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,      # let it grow as deep as needed
    class_weight=None,   # SMOTE took care of balancing
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

# Train both models on the SMOTE-balanced data
lr_sm.fit(X_train_sm, y_train_sm)
rf_sm.fit(X_train_sm, y_train_sm)

# Evaluate performance on original test set
evaluate_model("Logistic Regression + SMOTE", lr_sm, X_test_scaled, y_test)
evaluate_model("Random Forest + SMOTE", rf_sm, X_test_scaled, y_test)

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Helper function to calculate all the metrics we care about
def get_metrics(name, model, X_t, y_t):
    # Get predictions
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]  # probabilities for positive class

    # Return everything as a dictionary
    return {
        "Model": name,
        "Precision": precision_score(y_t, y_pred),
        "Recall": recall_score(y_t, y_pred),
        "F1": f1_score(y_t, y_pred),
        "ROC_AUC": roc_auc_score(y_t, y_proba)
    }

# Collect metrics for all models
rows = []

# Baseline models first
rows.append(get_metrics("LR baseline", lr, X_test_scaled, y_test))
rows.append(get_metrics("RF baseline", rf, X_test_scaled, y_test))

# Then the SMOTE versions
rows.append(get_metrics("LR + SMOTE", lr_sm, X_test_scaled, y_test))
rows.append(get_metrics("RF + SMOTE", rf_sm, X_test_scaled, y_test))

# Put everything into a dataframe for easy comparison
results_df = pd.DataFrame(rows)
print(results_df)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Function to visualize confusion matrix for any model
def plot_conf_mat(name, model, X_t, y_t):
    # Generate predictions
    y_pred = model.predict(X_t)

    # Build the confusion matrix
    cm = confusion_matrix(y_t, y_pred)

    # Create figure - keeping it compact
    plt.figure(figsize=(4, 3))

    # Draw heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    # Add labels and title
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()  # make sure nothing gets cut off
    plt.show()

# Plot confusion matrices for all our models
# Starting with baseline models
plot_conf_mat("LR baseline", lr, X_test_scaled, y_test)
plot_conf_mat("RF baseline", rf, X_test_scaled, y_test)

# Now the SMOTE-enhanced versions
plot_conf_mat("LR + SMOTE", lr_sm, X_test_scaled, y_test)
plot_conf_mat("RF + SMOTE", rf_sm, X_test_scaled, y_test)

from sklearn.metrics import roc_curve, auc

def plot_roc_two(name1, model1, name2, model2, X_t, y_t):
    # Get predicted probabilities for positive class
    y_proba1 = model1.predict_proba(X_t)[:, 1]
    y_proba2 = model2.predict_proba(X_t)[:, 1]

    # Calculate ROC curve points for both models
    fpr1, tpr1, thresholds1 = roc_curve(y_t, y_proba1)
    fpr2, tpr2, thresholds2 = roc_curve(y_t, y_proba2)

    # Compute area under curve
    auc_score1 = auc(fpr1, tpr1)
    auc_score2 = auc(fpr2, tpr2)


    plt.figure(figsize=(5, 4))

    # Plot first model ROC
    plt.plot(fpr1, tpr1, label=f"{name1} (AUC = {auc_score1:.3f})")

    # Plot second model ROC
    plt.plot(fpr2, tpr2, label=f"{name2} (AUC = {auc_score2:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – LR vs RF")  # Keeping original title
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

# Actually plot the comparison
plot_roc_two("LR baseline", lr, "RF baseline", rf, X_test_scaled, y_test)

import numpy as np

def plot_rf_feature_importance(model, feature_names, top_n=10):
    # Extract feature importances from the trained model
    importances = model.feature_importances_

    # Get indices of top N most important features (sorted ascending, so we take last N)
    indices = np.argsort(importances)[-top_n:]

    # Create the plot
    plt.figure(figsize=(6, 4))

    # Horizontal bar chart works better for feature names readability
    plt.barh(range(len(indices)), importances[indices], align="center")

    # Set y-axis labels to actual feature names
    feature_labels = [feature_names[i] for i in indices]
    plt.yticks(range(len(indices)), feature_labels)

    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Features – Random Forest")

    plt.tight_layout()
    plt.show()

# Plot the top 10 most important features
plot_rf_feature_importance(rf, X.columns, top_n=10)
     