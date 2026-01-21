import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data (hours → marks)
X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([12, 24, 36, 48, 60, 72, 84])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully")
