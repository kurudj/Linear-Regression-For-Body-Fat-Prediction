import numpy as np
import random
import csv
import math

def get_dataset(filename):
    """
    INPUT:
        filename - a string representing the path to the csv file.
    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        count = 1
        r = []
        newr = None
        for row in reader:
            r.append(row['BODYFAT'])
            r.append(row['DENSITY'])
            r.append(row['AGE'])
            r.append(row['WEIGHT'])
            r.append(row['HEIGHT'])
            r.append(row['ADIPOSITY'])
            r.append(row['NECK'])
            r.append(row['CHEST'])
            r.append(row['ABDOMEN'])
            r.append(row['HIP'])
            r.append(row['THIGH'])
            r.append(row['KNEE'])
            r.append(row['ANKLE'])
            r.append(row['BICEPS'])
            r.append(row['FOREARM'])
            r.append(row['WRIST'])
            dataset.append(r)
            r = []
    return dataset

def print_stats(dataset, col):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.
                  For example, 1 refers to density.
    RETURNS:
        None
    """
    print (len(dataset))
    sum = 0
    arr = []
    for datapoint in dataset:
        arr.append(float(datapoint[col]))
    print ("{:.2f}".format(np.mean(arr)))
    print ("{:.2f}".format(np.std(arr)))
    pass

def regression(dataset, cols, betas):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        mse of the regression model
    """
    n = len(dataset)
    if n==16:
        count = 0
        f = 0
        for i in range(len(betas)):
            if i==0:
                sum = betas[i]
            else:
                sum = betas[i]*float(dataset[cols[count]])
                count+=1
            f+=sum
        f-=float(dataset[0])
        return f** 2
    firstB = betas[0]
    mse = 0
    function = 0
    count = 0
    par = 0
    for row in dataset:
        count = 0
        for beta in betas:
            if beta==firstB:
                sum = beta
            else:
                sum = beta*float(row[cols[count]])
                count+=1
            par+=sum
        par-=float(row[0])
        function = par** 2
        par = 0
        mse+=function
    mse/=n
    return mse

def regression_gradient_wrtb(dataset, cols, betas, index):
    mse = 0
    for row in dataset:
        function=0
        par = 0
        sum = betas[0]
        for i in range(1, len(betas)):
            sum += betas[i]*float(row[cols[i-1]])
        sum-=float(row[0])
        function = sum
        if index!=0:
            function*=float(row[index])
        mse+=function
    mse/=252
    return mse*2

def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        An 1D array of gradients
    """
    grads = [0 for i in range(len(betas))]
    grads[0] = regression_gradient_wrtb(dataset, cols, betas, 0)
    count = 1
    for index in cols:
        grads[count] = regression_gradient_wrtb(dataset, cols, betas, index)
        count+=1
    return np.array(grads)

def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None
    """
    mse = 0
    for i in range(T):
        gradArr = gradient_descent(dataset, cols, betas)
        gradArr*=eta
        betas-=gradArr
        mse = regression(dataset, cols, betas)
        s = str(i+1) + " " + str("{:.2f}".format(mse))
        for j in range(len(betas)):
            s+=" "
            s+=str("{:.2f}".format(betas[j]))
        print(s)
    pass

def compute_betas(dataset, cols):
    """
    TODO: implement this function.
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = None
    subset = []
    column = [1 for i in range(len(dataset))]
    y = [k[0] for k in dataset]
    y = np.array(y,dtype = float)
    subset.append(column)
    for i in cols:
        column = [k[i] for k in dataset]
        subset.append(column)
    realsubset = np.transpose(subset)
    transpose = np.transpose(realsubset)
    finaltranspose = np.array(transpose,dtype = float)
    finalsubset = np.array(realsubset,dtype = float)
    firstSub = np.dot(finaltranspose,finalsubset)
    inverse = np.linalg.inv(firstSub)
    secondSub = np.dot(inverse,finaltranspose)
    betas = np.dot(secondSub,y)
    mse = regression(dataset, cols, betas)
    list = []
    list.append(mse)
    for beta in betas:
        list.append(beta)
    return tuple(list)

def predict(dataset, cols, features):
    """
    TODO: implement this function.
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values
    RETURNS:
        The predicted body fat percentage value
    """
    result = 0
    betas = compute_betas(dataset,cols)
    result+=betas[1]
    for i in range(2, len(betas)):
        result+=betas[i]*features[i-2]
    return result

def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)

def stochastic_gradient_descent(dataset, cols, betas, index):
    grads = [0 for i in range(len(betas))]
    function = betas[0]
    count = 0
    for i in range(1, len(betas)):
        function += betas[i]*float(dataset[index][cols[count]])
        count += 1
    function-=float(dataset[index][0])
    function*=2
    count = 0
    for colindex in cols:
        if count==0:
            grads[0] = function
            count+=1
        grads[count] = function*float(dataset[index][colindex])
        count+=1
    return np.array(grads, dtype = float)

def sgd(dataset, cols, betas, T, eta):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None
    """
    index = random_index_generator(0,252)
    for i in range(T):
        ind = next(index)
        gradArr = stochastic_gradient_descent(dataset, cols, betas, ind)
        gradArr*=eta
        betas-=gradArr
        mse = regression(dataset, cols, betas)
        s = str(i+1) + " " + str("{:.2f}".format(mse))
        for j in range(len(betas)):
            s+=" "
            s+=str("{:.2f}".format(betas[j]))
        print (s)
    pass
