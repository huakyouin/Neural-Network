import scipy.signal as signal
import numpy as np

def sech_square(x):
    return np.multiply(1/np.cosh(x),1/np.cosh(x))


def CNN_predict(w,X,kernel_size,nHidden,nLabels):
    nInstances,nVars = X.shape
    if kernel_size!=0:
        nVars=int((np.sqrt(nVars-1)-kernel_size+1)**2)

    # Form Weights
    kernelWeights=w[0:kernel_size*kernel_size].reshape((kernel_size,kernel_size),order='F')
    offset = kernel_size*kernel_size
    inputWeights = w[offset:offset+nVars*nHidden[0]].reshape((nVars,nHidden[0]),order='F')
    offset += nVars*nHidden[0]
    hiddenWeights=[]
    for h in range(1,len(nHidden)):
        hiddenWeights.append(w[offset:offset+nHidden[h-1]*nHidden[h]].reshape((nHidden[h-1],nHidden[h]),order='F'))
        offset = offset+nHidden[h-1]*nHidden[h]
    outputWeights = w[offset:offset+nHidden[-1]*nLabels]
    outputWeights = outputWeights.reshape((nHidden[-1],nLabels),order='F')

    # compute output
    Z=[]
    for i in range(nInstances):
        x=X[i,1:].reshape((16,16),order='F')
        conv= signal.convolve2d(x,kernelWeights,'valid')
        nW,nH=conv.shape
        conv=conv.reshape(nW*nH,order='F')
        Z.append(conv)
    Z=np.array(Z)
    ip=[]
    fp=[]
    ip.append(np.dot(np.atleast_2d(Z),inputWeights))  # 进入第一个隐藏层 实例数*firstN
    fp.append(np.tanh(ip[0]))   # 激活 实例数*firstN
    for h in range(1,len(nHidden)):  
        ip.append(np.dot(fp[h-1],hiddenWeights[h-1]))    #实例数*iN
        fp.append(np.tanh(ip[h]))       #实例数*iN
    yhat = np.dot(fp[-1],outputWeights)  # 模型估计各类型概率 实例数*类型数（10）
    y=np.argmax(yhat, axis=1)
    return y

def CNN_loss(w,X,y,kernel_size,nHidden,nLabels,Lossmethod='square'):
    X=np.atleast_2d(X)
    y=np.atleast_2d(y)
    nInstances,nVars = X.shape  # 实例个数，维度
    if kernel_size!=0:
        nVars=int((np.sqrt(nVars-1)-kernel_size+1)**2)

    # Form Weights
    kernelWeights=w[0:kernel_size*kernel_size].reshape((kernel_size,kernel_size),order='F')
    offset = kernel_size*kernel_size
    inputWeights = w[offset:offset+nVars*nHidden[0]].reshape((nVars,nHidden[0]),order='F')
    offset += nVars*nHidden[0]
    hiddenWeights=[]
    for h in range(1,len(nHidden)):
        hiddenWeights.append(w[offset:offset+nHidden[h-1]*nHidden[h]].reshape((nHidden[h-1],nHidden[h]),order="F"))
        offset = offset+nHidden[h-1]*nHidden[h]
    outputWeights = w[offset:offset+nHidden[-1]*nLabels]
    outputWeights = outputWeights.reshape((nHidden[-1],nLabels),order='F') # lastN*ylabelN

    gKernel= np.zeros((kernel_size,kernel_size))
    gInput = np.zeros((inputWeights.shape))
    gHidden=[]
    for h in range(1,len(nHidden)):
        gHidden.append(np.zeros((hiddenWeights[h-1].shape)))
    gOutput = np.zeros((outputWeights.shape))


    # Compute Output
    yhat=np.zeros((nInstances,outputWeights.shape[1]))
    Z=[]
    for i in range(nInstances):
        x=X[i,1:].reshape((16,16),order='F')
        conv= signal.convolve2d(x,kernelWeights,'valid')
        nW,nH=conv.shape
        conv=conv.reshape(nW*nH,order='F')
        Z.append(conv)
    Z=np.array(Z)
    ip=[]
    fp=[]

    ip.append(np.dot(np.atleast_2d(Z),inputWeights))  # 进入第一个隐藏层 实例数*firstN
    fp.append(np.tanh(ip[0]))   # 激活 实例数*firstN
    for h in range(1,len(nHidden)):  
        ip.append(np.dot(fp[h-1],hiddenWeights[h-1]))    #实例数*iN
        fp.append(np.tanh(ip[h]))       #实例数*iN
    yhat = np.dot(fp[-1],outputWeights)  # 模型估计各类型概率 实例数*类型数（10）
    if Lossmethod=='square':
        relativeErr = yhat-y
        f = np.sum(relativeErr**2)
        err = np.atleast_2d(2*relativeErr)  # 实例数*类型数 概率分布与真实的误差
    if Lossmethod=='softmax':
        yhat=yhat.T
        shift_x = yhat - np.max(yhat,axis=0)
        s=np.exp(shift_x) 
        s=(s/np.sum(s,axis=0)).T  
        f = np.multiply(-np.log(s),y).sum() # 前向总误差  
        err=s
        err[y==1]-=1
        
    
    gOutput = np.dot(fp[-1].T,err)/nInstances   #  lastN*(实例数*实例数)*类型数
    err=np.atleast_2d(np.multiply(sech_square(ip[-1]),np.dot(err,outputWeights.T)))  # 实例数*lastN
    ## n个隐藏层只有n-1片权重区
    for h in range(len(nHidden)-2,-1,-1):
        # hN*实例数*实例数*(h+1)N= hN*(h+1)N
        gHidden[h]=np.dot(fp[h].T,err)/nInstances
        # 实例数*(h+1)N*(h+1)N*hN=实例数*hN
        err=np.atleast_2d(np.multiply(sech_square(ip[h]),np.dot(err,hiddenWeights[h].T))) 
    gInput= np.dot(Z.T,err)/nInstances
    err=np.dot(err,inputWeights.T)
    for i in range(nInstances):
        error=err[i,:].reshape((nW,nH),order='F')
        reversed_X = X[i,:0:-1].reshape((16,16),order='F')
        if i==0:
            gconv=signal.convolve2d(X[i:,1:].reshape((16,16),order='F'),error,'valid')
        else:
            gconv+=signal.convolve2d(X[i:,1:].reshape((16,16),order='F'),error,'valid')
        # if i==0:
        #     gconv=signal.convolve2d(reversed_X,error,'valid')
        # else:
        #     gconv+=signal.convolve2d(reversed_X,error,'valid')
    gconv/=nInstances

    # Put Gradient into vector
    g = np.zeros((w.shape))
    # 输入到第一层
    g[0:kernel_size*kernel_size]=gconv.reshape((kernel_size*kernel_size,1),order='F')
    offset=kernel_size*kernel_size
    g[offset:offset+nVars*nHidden[0]] = gInput.reshape((nVars*nHidden[0],1),order='F')
    offset += nVars*nHidden[0]
    # 第一层到最后一层
    for h in range(1,len(nHidden)):
        g[offset:offset+nHidden[h-1]*nHidden[h]] = gHidden[h-1].reshape((nHidden[h-1]*nHidden[h],1),order='F')
        offset = offset+nHidden[h-1]*nHidden[h]
    # 最后一层到输出
    g[offset:offset+nHidden[-1]*nLabels] = gOutput.reshape((nHidden[-1]*nLabels,1),order='F')
    return f,g
