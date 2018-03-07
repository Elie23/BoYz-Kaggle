#test
import numpy   as np 
from scipy import misc
x_test = np.loadtxt("test_x.csv", delimiter=",")
print 'c long en ta...'
x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
print 'bar..'
y = np.loadtxt("train_y.csv", delimiter=",") 
print 'nak!'
x_test = x_test.reshape(-1,64,64)
x = x.reshape(-1, 64, 64) # reshape 
y = y.reshape(-1, 1) 
print x.reshape(-1,4096)

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
x = x.reshape(-1,4096)
y = y.reshape(50000)
# print x.reshape(-1,4096).shape, y.shape
for i in np.logspace(-5,5,11):
    C = i
    svm = LinearSVC(C=C)
    svm_fit = svm.fit(x, y).predict(x)
    print C, f1_score(y,svm_fit, average='macro')


    
# NEURAL NETWORKS POUR GROS GARS SEULEMENT    
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)


def NN_predict(x_train,y_train,step,epoch) :
    hiddenlayer_neurons=10
    output_neurons=10
    inputlayer_neurons = x_train.shape[1] #first layer corresponds to features
    wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh=np.random.uniform(size=(1,hiddenlayer_neurons))
    wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout=np.random.uniform(size=(1,output_neurons))
    
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input1=np.dot(x_train,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        pred = sigmoid(output_layer_input)
        
        #Backpropagation
        E = y_train-pred
        slope_output_layer = derivatives_sigmoid(pred)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) * step
        bout += np.sum(d_output, axis=0,keepdims=True) * step
        wh += x_train.T.dot(d_hiddenlayer) * step
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * step
    return pred
        

NN_pred = NN_predict(x,y,0.1,1)
