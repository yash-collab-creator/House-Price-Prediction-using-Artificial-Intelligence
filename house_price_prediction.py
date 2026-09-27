"""
Tiny AI Project: House Price Prediction
Uses Linear Regression from scikit-learn.
"""

from sklearn.linear_model import LinearRegression

# Training data: area (sq ft) -> price (₹ lakh)
X = [[500], [700], [900], [1100], [1300], [1500], [1800], [2200]]
y = [20, 28, 36, 44, 52, 60, 72, 88]

model = LinearRegression()
model.fit(X, y)

area = float(input("Enter house area in square feet: "))
prediction = model.predict([[area]])[0]

print(f"Estimated house price: ₹{prediction:.2f} lakh")
print("Note: This is a tiny educational model using a small sample dataset.")
