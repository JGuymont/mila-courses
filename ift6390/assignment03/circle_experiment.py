import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random
import time

from mlp.mlp import MLPClassifier
from mlp.dataloader import DataLoader

CIRCLE_DATA_PATH = './data/circles/circles.txt'

def plot_decision(X, y, title, mlp, param, ax=None, h=0.07):                    
    """plot the decision boundary. h controls plot quality."""              
    if ax is None:                                                          
        fig, ax = plt.subplots(figsize=(7, 6))                              
                                                                            
    # https://stackoverflow.com/a/19055059/6027071                          
    # sample a region larger than our training data X                      
    x_min = X[:, 0].min() - 0.5                                            
    x_max = X[:, 0].max() + 0.5                                            
    y_min = X[:, 1].min() - 0.5                                            
    y_max = X[:, 1].max() + 0.5                                            
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),                        
                            np.arange(y_min, y_max, h))                        
                                                                            
    # plot decision boundaries                                              
    x = np.concatenate(([xx.ravel()], [yy.ravel()]))                        
    pred = mlp.predict(x.T).reshape(xx.shape)                              
    ax.contourf(xx, yy, pred, alpha=0.8,cmap='RdYlBu')                      
                                                                            
    # plot points (coloured by class)                                      
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, cmap='RdYlBu')            
    ax.axis('off')
    plt.title('hidden_dim: {} | learning rate: {} | n_epochs: {} | lambda_1: {} | lambda_2: {}'.format(
            param[0], param[1], param[2], param[3], param[4]
        )
    )                                                          
                                                                            
    plt.savefig(title)                                                    
    plt.close()

class Data:
    """Abstract class for the circles dataset

    Args
        path: (string) path to the dataset
        input_dim: (int)
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """

    def __init__(self, path, input_dim, split):
        self._raw_data = np.loadtxt(open(path, 'r'))
        self._input_dim = input_dim
        self._data = self._read_data()
        self._data_index = self._split_index(split)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def _read_data(self):
        inputs = self._raw_data[:, :self._input_dim]
        targets = self._raw_data[:, -1].astype(int)
        return list(zip(inputs, targets))

    def _split_index(self, split):
        storage = {'train': [], 'valid': [], 'test': []}
        train_size = round(len(self)*split[0])
        valid_size = round((len(self) - train_size)*split[1])

        examples = range(len(self))
        storage['train'] = random.sample(examples, train_size)
        examples = [ex for ex in examples if ex not in storage['train']] # remove index
        storage['valid'] = random.sample(examples, valid_size)
        storage['test'] = [ex for ex in examples if ex not in storage['valid']]
        return storage

    def train(self):
        return [self._data[i] for i in self._data_index['train']]

    def valid(self):
        return [self._data[i] for i in self._data_index['valid']]

    def test(self):
        return [self._data[i] for i in self._data_index['test']]

    def dim_(self):
        return self._input_dim

def plot_gradient(grad1, grad2, param_names, legend, title):
    plt.rcParams.update({'font.size': 6})
    plt.plot(param_names, grad1, '--')
    plt.plot(param_names, grad2, 'o')
    plt.legend(legend)
    plt.xlabel('parameter')
    plt.ylabel('gradient')
    plt.savefig(title)
    plt.show()

if __name__ == '__main__':

    INPUT_DIM = 2
    OUTPUT_DIM = 2

    data = Data(CIRCLE_DATA_PATH, input_dim=2, split=[0.7, 0.15, 0.15])

    def question12():
        batch = DataLoader(data.train(), batch_size=1)
        X = batch[0][0]
        Y = batch[0][1]
        
        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        gradHats, grads, param_names = mlp.finite_difference_check(X, Y, crazy_loop=True)

        plot_gradient(
            gradHats, grads, 
            param_names, 
            legend=['finite differences approx.', 'backpropagation'],
            title='plots/question2.jpg'
        )
        
    def question4():
        batch = DataLoader(data.train(), batch_size=10)
        X = batch[20][0]
        Y = batch[20][1]

        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        gradHats, grads, param_names = mlp.finite_difference_check(X, Y, crazy_loop=True)

        plot_gradient(
            gradHats, grads, 
            param_names, 
            legend=['finite differences approx.', 'backpropagation'],
            title='plots/question4.jpg'
        )

    def question5():

        raw_data = np.loadtxt(open(CIRCLE_DATA_PATH, 'r'))
        X = raw_data[:, :2]
        y = raw_data[:, -1]


        # hyperparameters
        BATCH_SIZE = 32

        HIDDEN_DIM_SET = [2, 4, 6, 8]
        NUM_EPOCH_SET = [50, 100]
        LEARNING_RATE_SET = [0.05]
        L1_WEIGH_DECAY = [0, 0.1]
        L2_WEIGH_DECAY = [0, 0.1]
        
        trainloader = DataLoader(data.train(), batch_size=BATCH_SIZE)
        devloader = DataLoader(data.valid(), batch_size=len(data.valid()))

        i = 0
        for h in HIDDEN_DIM_SET:
            for lr in LEARNING_RATE_SET:
                for l1 in L1_WEIGH_DECAY:
                    for l2 in L2_WEIGH_DECAY:
                        for n_epoch in NUM_EPOCH_SET:
                
                            print('\nhidden_dim: {}, lr: {}, l1: {}, l2: {}'.format(h, lr, l1, l2))
                            mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM, h, lr, n_epoch, l1, l2, l1, l2)
                            mlp.train(trainloader, devloader, crazy_loop=False)

                            plot_decision(X, y, './plots/question5{}.jpg'.format(i), mlp, param=[h, lr, n_epoch, l1, l2, l1, l2])
                            i += 1       

    def question7():

        trainloader = DataLoader(data.train(), batch_size=1)

        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        
        mlp.backward(*trainloader[0], crazy_loop=False)
        grads, param_names = mlp.get_gradients()

        mlp.backward(*trainloader[0], crazy_loop=True)
        grad_loop, param_names = mlp.get_gradients()

        plot_gradient(
            grads, grad_loop, 
            param_names, 
            legend=['Matrix calculus', 'Loop'],
            title='plots/question71.jpg'
        )

        trainloader = DataLoader(data.train(), batch_size=10)
        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        
        mlp.backward(*trainloader[0], crazy_loop=False)
        grads, param_names = mlp.get_gradients()

        mlp.backward(*trainloader[0], crazy_loop=True)
        grad_loop, param_names = mlp.get_gradients()

        plot_gradient(
            grads, grad_loop, 
            param_names, 
            legend=['Matrix calculus', 'Loop'],
            title='plots/question72.jpg'
        )

    def question8():
        trainloader = DataLoader(data.train(), batch_size=32)
        devloader = DataLoader(data.valid(), batch_size=len(data.valid()))

        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM, 10, 0.05, 1000)
        mlp.train(trainloader, devloader, crazy_loop=False)

        mlp.initialize()
        print('-----------------------------------')

        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM, 10, 0.05, 1000)
        mlp.train(trainloader, devloader, crazy_loop=True)
    

    #question12()
    #question4()
    #question5()
    #question7()
    question8()