
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Make results more reproducible
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the preprocessed data
data = np.load("processed_data.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 2. Build the ANN
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# 3. Configure the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# 4. Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 5. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Evaluate on the separate test set
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

# 7. Generate predictions
probabilities = model.predict(X_test, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)

print("\nTest loss:", round(float(test_loss), 4))
print("Test accuracy:", round(float(test_accuracy), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "\nROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 4)
)

# 8. Save the trained model
model.save("churn_ann.keras")

# Save training history for plotting later
joblib.dump(history.history, "training_history.joblib")

print("\nModel saved as churn_ann.keras")
