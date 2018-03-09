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
from sklearn.preprocessing import StandardScaler

#DATA SPLIT BOIZ
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)
print x_train.shape, y_train.shape, x_val.shape, y_val.shape

#Preprocessing
scaler = StandardScaler().fit(x_train)

# Scale the train set
x_train = scaler.transform(x_train)

# Scale the validation set
x_val = scaler.transform(x_val)

#SVM
def SVM(x_train, y_train, x_val, y_val, C):
    svm = LinearSVC(C=C)
    svm_fit = svm.fit(x_train, y_train).predict(x_val)
    return f1_score(y_val,svm_fit, average='macro')

#Find the best C
for i in np.linspace(1e-6,1e-5,10):
    print SVM(x_train, y_train, x_val, y_val, i)

    
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
    #print(wout)
    
    for i in range(epoch):
        #Forward Propogation
        print i
        hidden_layer_input = np.matmul(x_train,wh) + bh
        print np.argmax(hidden_layer_input), np.ravel(hidden_layer_input)[np.argmax(hidden_layer_input)]
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input = np.matmul(hiddenlayer_activations,wout) + bout
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
        
        #F1-score
        f1 = NN_validation(x_val, y_val, wout, bout, wh, bh)
        print "step:", i, "f1 on validation data:", f1
    return pred,output_layer_input, wout, bout, wh, bh

def NN_validation(x_val, y_val, wout, bout, wh, bh):
    hidden_layer_input = np.matmul(x_val,wh) + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input = np.matmul(hiddenlayer_activations,wout) + bout
    pred = sigmoid(output_layer_input)
    return f1_score(y_val,np.argmax(pred, axis = 1), average='macro')

NN_pred, output_layertrain, wout, bout, wh, bh = NN_predict(x_train,y_train, x_val, y_val, 1e-5, 100)



#Version qui change y_train et y_val pour etre 40000x10 au lieu de 40000x1 dans le NN
# NEURAL NETWORKS QUI MARCHE PAS    
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)


def NN_predict(x_train,y_trainNN,x_val,y_valNN,step,epoch) :
    hiddenlayer_neurons=50
    output_neurons=10
    inputlayer_neurons = x_train.shape[1] #first layer corresponds to features
    wh=2*np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))-1
    bh=2*np.random.uniform(size=(1,hiddenlayer_neurons))-1
    wout=2*np.random.uniform(size=(hiddenlayer_neurons,output_neurons))-1
    bout=2*np.random.uniform(size=(1,output_neurons))-1
    #print(wout)
    
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input = np.matmul(x_train,wh) + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input = np.matmul(hiddenlayer_activations,wout) + bout
        pred = sigmoid(output_layer_input)
        
        #Backpropagation
        E = y_trainNN-pred
        slope_output_layer = derivatives_sigmoid(pred)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = np.matmul(d_output,wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) * step
        bout += np.sum(d_output, axis=0,keepdims=True) * step
        wh += x_train.T.dot(d_hiddenlayer) * step
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * step
        
        #F1-score
        f1 = NN_validation(x_val, y_val, wout, bout, wh, bh)
        print "step:", i, "f1 on validation data:", f1
    return pred,output_layer_input, wout, bout, wh, bh

def NN_validation(x_val, y_val, wout, bout, wh, bh):
    hidden_layer_input = np.matmul(x_val,wh) + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input = np.matmul(hiddenlayer_activations,wout) + bout
    pred = sigmoid(output_layer_input)
    print y_val, np.argmax(pred, axis = 1)
    return f1_score(y_val,np.argmax(pred, axis = 1), average='macro')

y_trainNN = np.zeros(y_train.shape[0]*10).reshape(y_train.shape[0],10)
y_valNN = np.zeros(y_val.shape[0]*10).reshape(y_val.shape[0],10)


for i in range(y_train.shape[0]):
    y_trainNN[i][int(y_train[i])] = 1
    
for i in range(y_val.shape[0]):
    y_valNN[i][int(y_val[i])] = 1  
    
print y_trainNN.shape

NN_pred, output_layertrain, wout, bout, wh, bh = NN_predict(x_train,y_trainNN, x_val, y_valNN, 1e-4, 100)



#2 HIDDEN LAYERS

### NEURAL NETWORKS QUI MARCHE PAS    
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)


def NN_predict(x_train,y_trainNN,y_train,x_val,y_valNN,y_val,step,epoch,l2) :
    hiddenlayer_neurons = 100
    output_neurons = 10
    inputlayer_neurons = x_train.shape[1] #first layer corresponds to features
    wh1=2*np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))-1
    bh1=2*np.random.uniform(size=(1,hiddenlayer_neurons))-1
    wh2=2*np.random.uniform(size=(hiddenlayer_neurons,hiddenlayer_neurons))-1
    bh2=2*np.random.uniform(size=(1,hiddenlayer_neurons))-1
    wout=2*np.random.uniform(size=(hiddenlayer_neurons,output_neurons))-1
    bout=2*np.random.uniform(size=(1,output_neurons))-1
    #print(wout)
    
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input1 = np.matmul(x_train,wh1) + bh1
        hiddenlayer_activations1 = sigmoid(hidden_layer_input1)
        
        hidden_layer_input2 = np.matmul(hiddenlayer_activations1,wh2) + bh2
        hiddenlayer_activations2 = sigmoid(hidden_layer_input2)
        
        output_layer_input = np.matmul(hiddenlayer_activations2,wout) + bout
        pred = sigmoid(output_layer_input)
        
        #Backpropagation
        #print y_trainNN[:5], pred[:5]
        E = (y_trainNN-pred)
        #print E[:5]
        slope_output_layer = derivatives_sigmoid(pred)
        slope_hidden_layer1 = derivatives_sigmoid(hiddenlayer_activations1)
        slope_hidden_layer2 = derivatives_sigmoid(hiddenlayer_activations2)
        d_output = E * slope_output_layer

        Error_at_hidden_layer = np.matmul(d_output,wout.T)
        d_hiddenlayer1 = Error_at_hidden_layer * slope_hidden_layer1
        d_hiddenlayer2 = Error_at_hidden_layer * slope_hidden_layer2
        wout = wout - l2*wout 
        wout = hiddenlayer_activations2.T.dot(d_output) * step + wout
        bout = np.sum(d_output, axis=0,keepdims=True) * step + bout
        wh1 = wh1 - l2*wh1
        wh1 = x_train.T.dot(d_hiddenlayer1) * step + wh1
        bh1 = np.sum(d_hiddenlayer1, axis=0,keepdims=True) * step + bh1
        wh2 = wh2 - l2*wh2
        wh2 = hiddenlayer_activations1.T.dot(d_hiddenlayer2) * step + wh2
        bh2 = np.sum(d_hiddenlayer2, axis=0,keepdims=True) * step + bh2
        #print y_train[:5],np.argmax(pred,axis=1)[:5]
        #print (y_train==np.argmax(pred,axis=1)).sum()
        #F1-score
        f1val = NN_validation(x_val, y_val, wout, bout, wh1, bh1, wh2, bh2)
        f1train = precision_score(y_train, np.argmax(pred, axis = 1), average='macro') #f1_score(y_train,np.argmax(pred, axis = 1), average='macro')
        print "step:", i, "Error :", (0.5*E**2).sum(), "f1 on validation data:", f1val, "on train :", f1train
    return pred,output_layer_input, wout, bout, wh1, bh1, wh2, bh2

def NN_validation(x_val, y_val, wout, bout, wh1, bh1, wh2, bh2):
    hidden_layer_input1 = np.matmul(x_val,wh1) + bh1
    hiddenlayer_activations1 = sigmoid(hidden_layer_input1)
    
    hidden_layer_input2 = np.matmul(hiddenlayer_activations1,wh2) + bh2
    hiddenlayer_activations2 = sigmoid(hidden_layer_input2)
        
    output_layer_input = np.matmul(hiddenlayer_activations2,wout) + bout
    pred = sigmoid(output_layer_input)

    #print y_val, np.argmax(pred, axis = 1)
    return precision_score(y_val, np.argmax(pred, axis = 1), average='macro') #f1_score(y_val,np.argmax(pred, axis = 1), average='macro')

NN_pred, output_layertrain, wout, bout, wh1, bh1, wh2, bh2, wh3, bh3 = NN_predict(x_train,y_trainNN, y_train, x_val, y_valNN, y_val, 1e-5, 10000, 0)
