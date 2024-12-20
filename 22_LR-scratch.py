import numpy as np;

np.random.seed(42)

class CustomLinearRegression:
    def _init_(self,alpha=0.0001,epoch=10):
        self.alpha=alpha
        self.epoch=epoch
        self.w=np.random.random()
        self.b=np.random.random()


    def fit(self, X_train,y_train):
        self.num_rec=X_train.shape[0]
        for i in range(self.epoch):
            y_hat=self.w
            loss=y_hat
            grad_w=(2/self.num_rec) * np.sum(loss*X_train)
            grad_b=(2/self.num_rec) * np.sum(loss)

            self.w=self.w-self.alpha*grad_w
            self.b=self.b-self.alpha*grad_b
def predict(self,X_test):
    return self.w*X_test+self.b
       
      