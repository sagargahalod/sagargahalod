import numpy

def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))

def error(predict,target):
    return numpy.power(predict-target,2)

def err_pred_der(predict,target):
    return 2*(predict-target)

def act_der_sop(sop):
    return sigmoid(sop) * (1-sigmoid(sop))

def sop_w_der(x):
    return x

def update_w(w,grad,learning_rate):
    return w-learning_rate*grad

x=0.1
target=0.3
learning_rate = 0.01
w=numpy.random.rand()

print("initial w ",w)

for k in range(10000):#keep increasing the weight to reach the target going from 0.1->0.3
    y=w*x
    predicted=sigmoid(y)
    err=error(predicted,target)

    g1=err_pred_der(predicted,target)
    g2=act_der_sop(predicted)
    g3=sop_w_der(x)

    grad=g3 * g2 * g1

    print(predicted)

    w=update_w(w,grad,learning_rate)