#test
import numpy   as np 
from scipy import misc
x_test = np.loadtxt("test_x.csv", delimiter=",")
print 'c long en ta...'
x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
print 'bar..'
y = np.loadtxt("train_y.csv", delimiter=",") 
print 'nak!'

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#DATA SPLIT BOIZ
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)
print x_train.shape, y_train.shape, x_val.shape, y_val.shape

#GRINDING
for i in np.logspace(-10,-6,5):
    C = i
    print C
    svm = LinearSVC(C=C)
    svm_fit = svm.fit(x_train, y_train).predict(x_val)
    print C, f1_score(y_val,svm_fit, average='macro')


    
# NEURAL NETWORKS QUI MARCHE PAS    
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)


def NN_predict(x_train,y_train,step,epoch) :
    y_train = np.resize(y_train,(len(y_train),1))
    hiddenlayer_neurons=100
    output_neurons=10
    inputlayer_neurons = x_train.shape[1] #first layer corresponds to features
    wh=2*np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))-1
    bh=2*np.random.uniform(size=(1,hiddenlayer_neurons))-1
    wout=2*np.random.uniform(size=(hiddenlayer_neurons,output_neurons))-1
    bout=2*np.random.uniform(size=(1,output_neurons))-1
    print(wout)
    
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input1=np.matmul(x_train,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.matmul(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        pred = sigmoid(output_layer_input)
        
        #Backpropagation
        E = y_train-pred
        slope_output_layer = derivatives_sigmoid(pred)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = np.matmul(d_output,wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) * step
        bout += np.sum(d_output, axis=0,keepdims=True) * step
        wh += x_train.T.dot(d_hiddenlayer) * step
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * step
    return pred,output_layer_input
        

NN_pred = NN_predict(x2,y2,0.1,100)

