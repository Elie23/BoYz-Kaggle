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
