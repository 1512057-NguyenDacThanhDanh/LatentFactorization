import numpy as np
import pandas as pd
import csv

class MFF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        http://www.albertauyeung.com/post/python-matrix-factorization/?fbclid=IwAR0vqbNaSjUqYUeGLV9GeXxW5Vx9ZMyLl_3HaWFSZJ5nJ5TxB9A4DOJK1zY

        - b  is the global bias (which can be easily estimated by using the mean of all ratings), 
        - b_u[i] is the bias of user i
        - b_i[j] is te bias of item j.
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        #training_process = []
        new_mse = 1
        old_mse = 1
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
          
            self.sgd()
            if (i == 0): 
                predicted = self.full_matrix()              
                new_mse = self.mse(predicted)
                old_mse = new_mse
            else:                
                new_mse = self.mse(predicted)                
                if (new_mse <= old_mse):
                    predicted = self.full_matrix()
                    old_mse = new_mse
            #training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, old_mse))

        #return training_process

    def mse(self, predicted):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        #predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        # E = ∑(r[i,j] - P[i,k] * Q[k,j]) ^ 2
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = 2 * (r - prediction)

            '''
                b_u′[i] = b_u[i] + alpha × (eij − beta * b_u[i])
                b_i′[j] = b_i[j] + alpha × (eij − beta * b_i[j])
            '''
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            '''
                # P'[i,k] = P[i,k] + alpha * (2 * e * Q[k,j] - beta * P[i,k])
                # Q'[k,j] = Q[k,j] + alpha * (2 * e * P[i,k] - beta * Q[k,j])
            '''
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])


    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        
        Full = P * Q.T

        Full = bias + P * Q.T

        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis,:] + self.P.dot(self.Q.T) 


dataset = pd.read_csv('./Matrix3.csv', sep=',').values
mfd = dataset[:,1:]
dataset = dataset[0:0]
with open('./Matrix3.csv') as csv_file:
    d = [row for row in csv.reader(csv_file)]
    print('\n\n')

mf = MFF(mfd, K=10, alpha=0.005, beta=0.01, iterations=100)
mf.train()
mfd = mf.full_matrix()

row = len(d[1:]) + 1
for x in range(1, row):
    d[x][1:]=mfd[x-1]
        
with open('./FullMatrix3_2.csv', mode='w', newline='') as csv_file:
    w = csv.writer(csv_file)
    for row in d:
        w.writerow(row)
    print('Completed')