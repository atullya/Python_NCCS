import numpy as np
import pandas as pd

# class CustomLinearRegression:
#     def __init__(self, alpha=0.0001, n_iter=10):  # alpha -> learning rate
#         self.alpha = alpha
#         self.n_iter = n_iter

#     def train(self, X, y):
#         self.n_rec, self.n_features = X.shape

#         # Convert to numpy array if X or y is a pandas DataFrame/Series
#         if isinstance(X, (pd.DataFrame, pd.Series)):
#             X = X.to_numpy()
#         if isinstance(y, pd.Series):
#             y = y.to_numpy()
#         elif isinstance(X, np.ndarray):
#             pass
#         else:
#             raise Exception("X should be either pandas DataFrame or numpy array")

#         # Initialize weights and bias
#         self.w = np.random.random(self.n_features)
#         self.b = np.random.random()

#         for i in range(self.n_iter):
#             print(f"Epoch: {i + 1}")

#             # Prediction
#             y_hat = self.predict(X)

#             # Difference for gradient calculation
#             diff = y_hat - y

#             # Loss calculation
#             loss = self.loss_mse(y, y_hat)

#             # Gradient calculation
#             grad_b = (2 / self.n_rec) * np.sum(diff)
#             grad_w = (2 / self.n_rec) * np.dot(diff, X)

#             # Updating weights and bias
#             self.b -= self.alpha * grad_b
#             self.w -= self.alpha * grad_w

#             print(f"grad_b: {grad_b}, grad_w: {grad_w}, b: {self.b}, w: {self.w}")
#             print(f"Loss: {loss}")
#             print("=====================================")
            
#     def predict(self, X):
#         # Convert to numpy array if X is a pandas DataFrame/Series
#         if isinstance(X, (pd.DataFrame, pd.Series)):
#             X = X.to_numpy()
#         elif isinstance(X, np.ndarray):
#             pass
#         else:
#             raise Exception("X should be either pandas DataFrame or numpy array")

#         # Prediction for multiple features: y = w1*x1 + w2*x2 + ... + b
#         return np.dot(X, self.w) + self.b
    
#     def loss_mse(self, y, y_hat):
#         # Mean Squared Error (MSE)
#         return np.mean((y - y_hat) ** 2)


# Example usage
# Assuming X_train is a pandas DataFrame or numpy array and y_train is a pandas Series or numpy array
# custome_lr = CustomLinearRegression(alpha=0.01, n_iter=100)
# custome_lr.train(X_train[['X2 house age', 'X5 latitude']], y_train)



class CustomLinearRegression1:
    def __init__(self, alpha=0.0001, epoch=10):  # epoch -> fancy word for iteration
        self.alpha = alpha
        self.epoch = epoch
        # self.w1 = np.random.random()
        # self.w2 = np.random.random()
        self.b = np.random.random()
    
    def fit(self, X_train, y_train):
        self.num_rec, self.n_features = X_train.shape  # number of records -> (data, features)
        self.w= np.random.random(self.n_features)
        for i in range(self.epoch):
            # Predictions
            y_hat = self.predict(X_train)
            
            # How far the predictions (y_hat) are from ground truth (y_train)
            loss = y_hat - y_train
            
            # Gradient calculation
            grad_w1 = (2 / self.num_rec) * np.sum(loss * X_train['X2 house age'])
            grad_w2 = (2 / self.num_rec) * np.sum(loss * X_train['X5 latitude'])
            grad_b = (2 / self.num_rec) * np.sum(loss)
            
            # Updating
            self.w1 = self.w1 - self.alpha * grad_w1
            self.w2 = self.w2 - self.alpha * grad_w2
            self.b = self.b - self.alpha * grad_b
        
        return self
      
    def predict(self, X):
        return (
            self.w1 * X['X2 house age'] + 
            self.w2 * X['X5 latitude'] + 
            self.b
        )
class CustomLinearRegression2:
    def __init__(self, alpha=0.0001, epoch=10):  # epoch -> fancy word for iteration
        self.alpha = alpha
        self.epoch = epoch
        self.w1 = np.random.random()
        self.w2 = np.random.random()
        self.b = np.random.random()
    
    def fit(self, X_train, y_train):
        self.num_rec = X_train.shape[0]  # number of records -> (data, features)
        
        for i in range(self.epoch):
            # Predictions
            y_hat = self.predict(X_train)
            
            # How far the predictions (y_hat) are from ground truth (y_train)
            loss = y_hat - y_train
            
            # Gradient calculation
            grad_w1 = (2 / self.num_rec) * np.sum(loss * X_train['X2 house age'])
            grad_w2 = (2 / self.num_rec) * np.sum(loss * X_train['X5 latitude'])
            grad_b = (2 / self.num_rec) * np.sum(loss)
            
            # Updating
            self.w1 = self.w1 - self.alpha * grad_w1
            self.w2 = self.w2 - self.alpha * grad_w2
            self.b = self.b - self.alpha * grad_b
        
        return self
      
    def predict(self, X):
        return (
            self.w1 * X['X2 house age'] + 
            self.w2 * X['X5 latitude'] + 
            self.b
        )


class CustomLinearRegression:
    def __init__(self,alpha=0.0001,epoch=10,n_iter=10):  # epoch -> fancy word for iteration. alpha -> learning rate
        self.alpha = alpha
        self.n_iter=n_iter
        self.by=np.random.random()

    def _feature_scaling(self,X):
        """Feature scaling for faster convergence"""
        self.feature_mean=np.mean(X,axis=0)
        self.feature_stds=np.std(X,axis=0)
        self.feature_stds[self.feature_stds==0]=1.0
        return (X-self.feature_mean)/self.feature_stds

    def train(self, X, y):
        self.n_rec, self.n_features=X.shape

        if(isinstance,(pd.DataFrame,pd.Series)) or isinstance(y,pd.Series):
            X=X.to_numpy()
        elif isinstance(X,np.ndarray):
            pass
        else:
            raise Exception("X should be either pandas dataframe or numpy array")
            
        self.w=np.random.random(self.n_features)
        self.b=np.random.random()

        for i in range(self.n_iter):
            print(f"Epoch:{i+1}")
            y_hat=self.predict(X) #prediction

            diff=y_hat-y-y #difference for gradient calculation
            loss=self.loss_mse(y,y_hat) #loss calculation

            #gradient calculation
            grad_b=-(2/self.n_rec)*np.sum(diff)
            grad_w=-(2/self.n_rec)*np.dot(diff,X)

            #updating
            self.b=self.b-self.alpha*grad_b
            self.w=self.w-self.alpha*grad_w

            print(f"grad b: {grad_b}, grad w: {grad_w}, b: {self.b}, w: {self.w}")
            print(f"Loss: {loss}")
            print("=====================================")
            
    def predict(self, X):  #for single feature i.e y=w*x+b
        if(isinstance,(pd.DataFrame,pd.Series)) or isinstance(y,pd.Series):
            X=X.to_numpy()
        elif isinstance(X,np.ndarray):
            pass
        else:
            raise Exception("X should be either pandas dataframe or numpy array")
        return np.dot(X,self.w)+self.b # for multiple features i.e y=w1*x1+w2*x2+...+b yesto ko lagi
    
    def loss_mse(self, y, y_hat):
        return np.mean((y-y_hat)**2)
