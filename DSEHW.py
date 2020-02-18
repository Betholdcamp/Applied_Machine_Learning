import numpy as np
    
class Perceptron(object):
    def __init__(self, learning_rate, random_state, initial_weight):
        self.learning_rate = learning_rate
        self.n_iter = 10
        self.random_state = random_state
        self.initial_weight = initial_weight
            
    def fit(self, X, y):
        print('CLASS A')
        self.errors_ = []
        rgen = np.random.RandomState(self.random_state)
        if self.initial_weight == None:
            self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
            
        else:
            self.w_ = list(initial_weight)
        
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self        
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
class Logistic(object):
    def __init__(self, learning_rate, random_state, initial_weight):
        self.learning_rate = learning_rate
        self.n_iter = 10
        self.random_state = random_state
        self.initial_weight = initial_weight
        print('CLASS B')
    
    def fit(self, X, y):
        print('CLASS B')
        self.cost_ = []
        rgen = np.random.RandomState(self.random_state) 
        if self.initial_weight == None:
            self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
            
        else:
            self.w_ = list(initial_weight)
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.learning_rate * X.T.dot(errors)
            self.w_[0] += self.learning_rate * errors.sum()
            #delta_ = self.learning_rate * X.T.dot(errors)
            #self.w_ = self.learning_rate * errors.sum() + delta_
            cost = (-y.dot(np.log(output)) -
                        ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
        
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

class HyperTan(object):
    def __init__(self, learning_rate, random_state, initial_weight):
        self.learning_rate = learning_rate
        self.n_iter = 10
        self.random_state = random_state
        self.initial_weight = initial_weight
        print('CLASS C')
    
    def fit(self, X, y):
        print('CLASS C')
        self.cost_ = []
        rgen = np.random.RandomState(self.random_state)     
        if self.initial_weight == None:
            self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
            
        else:
            self.w_ = list(initial_weight)
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.learning_rate * X.T.dot(errors)
            self.w_[0] += self.learning_rate * errors.sum()
            #delta_ = self.learning_rate * X.T.dot(errors)
            #self.w_ = self.learning_rate * errors.sum() + delta_
            cost = (-y.dot(np.log(output)) -
                        ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def activation(self, z):
        """Compute tangent activation"""
        return np.tanh(z)
        
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
'''    
class DSELinearClassifier(A, B, C):
    def __init__(self, activation, learning_rate, random_state, initial_weight):
        self.learning_rate = learning_rate
        self.n_iter = 10
        self.random_state = random_state
        self.initial_weight= initial_weight
        '''
        
        
            
            
            