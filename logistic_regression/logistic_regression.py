import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.002, num_iterations=100, momentum = 0.9):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.momentum = momentum
        
    @staticmethod
    def sigmoid(x):
        return 0.5 + 0.197*x - 0.004*(x**3)

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        _, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        velocity = np.zeros_like(self.weights)
        
        for _ in range(self.num_iterations):
            lookahead_weights = self.weights + self.momentum * velocity
            dot_X_W = np.dot(X, lookahead_weights)
            y_predict = self.sigmoid(dot_X_W)
            gradient =  np.dot((y_predict - y), X)
            velocity = self.momentum * velocity - self.learning_rate * gradient
            self.weights += velocity
            
    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        linear_model = np.dot(X, self.weights) 
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        linear_model = np.dot(X, self.weights)
        y_predicted = self.sigmoid(linear_model)
        return y_predicted 