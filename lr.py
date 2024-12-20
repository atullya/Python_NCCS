# ctrl + shift + p -> python interpreter
import numpy as np

np.random.seed(42)

class CustomLinearRegression:
    def __init__(self, alpha = 0.0001, epoch = 10): # epoch -> fancy word for iteration
        self.alpha = alpha
        self.epoch = epoch
        self.w = np.random.random()
        self.b = np.random.random()
        
    def fit(self, X_train, y_train):
        self.num_rec = X_train.shape[0] # number of records -> (data, features)
        
        for i in range(self.epoch):
            # preictions
            y_hat = self.predict(X_train)
            
            # how far the predictions (y_hat) are from ground truth (y_train)
            loss = y_hat - y_train
            
            # gradient calculation
            grad_w = (2/ self.num_rec) * np.sum(loss * X_train)
            grad_b = (2/ self.num_rec) * np.sum(loss)
            
            # updatings
            self.w1 = self.w1 - self.alpha * grad_w1
            self.w2 = self.w2 - self.alpha * grad_w2
            self.b = self.b - self.alpha * grad_b
        
        return self
        
    def predict(self, X):
        return self.w * X + self.b