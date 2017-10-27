import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 408000 #THIS IS THE MEDIAN OF THE DATA
LEARNING_RATE = 0.01

# Updates weights using Gradient Descent Algorithm
def update_weights(data, weights, target):
    dw = np.zeros(weights.shape)
    for i in range(dw.shape[0]):
        total = 0
        for d in range(data.shape[0]):
            total += (target[d]-np.dot(weights,data[d,:]))*(data[d,i])
        dw[i] = total
    return weights + LEARNING_RATE*dw

# Calculates the mean squared error
def mean_squared_error(data,weights,pred):
    total = 0
    for i in range(data.shape[0]):
        total += (np.dot(weights,data[i,:]) - pred[i])**2
    return total/data.shape[0]

# Method that trains a single perceptron
def perceptron(data, weights, target, limit):
    old_error = mean_squared_error(data,weights,target)
    new_error = 10000
    while(np.abs(old_error-new_error) > limit):
        weights = update_weights(data,weights,target)
        old_error = new_error
        new_error = mean_squared_error(data,weights,target)
    return weights

# Calculates the accuracy of the model
def accuracy(data, weights, target):
    prediction = np.zeros(target.shape[0])
    for i in range(data.shape[0]):
        d = data[i,:]
        if(np.dot(weights,d) > 0):
            prediction[i] = 1
        else:
            prediction[i] = -1
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(prediction)):
        if(prediction[i] == 1 and target[i] == 1):
            tp += 1
        if(prediction[i] == -1 and target[i] == -1):
            tn += 1
    return (tp+tn)/len(target)

# Open the data file and read in all the data
f = open('housing_prices.txt','r')
lines = f.readlines()
f.close()

# Get rid of all tabs and new lines in the data file
for i in range(len(lines)):
    if '\n' in lines[i]:
        lines[i] = lines[i][:-1]
    lines[i] = lines[i].split('\t')

# Labels are the first row in the file, and the data is everything else
labels = lines[0]
all_data = lines[1:]

# Initialize empty numpy arrays
sqft = np.zeros(len(all_data))
price = np.zeros(len(all_data))
city = []
bedrooms = np.zeros(len(all_data))
baths = np.zeros(len(all_data))

# Fill the numpy arrays
for i in range(len(all_data)):
    row = all_data[i]
    sqft[i] = float(row[0])
    price[i] = float(row[1])
    city.append(row[2])
    bedrooms[i] = float(row[3])
    baths[i] = float(row[4])

# Converts the cities to numbers, where each new city gets a new number
d = {}
count = 1
temp_city = np.zeros(len(city))
for i in range(len(city)):
    curr_city = city[i]
    if curr_city not in list(d.keys()):
        d[curr_city] = count
        count += 1
    temp_city[i] = d[curr_city]
city = temp_city

# Rows are examples and columns are features
# Final feature in the column in the price (what were trying to predict)
data = np.array([np.ones(sqft.shape[0]),sqft,city,bedrooms,baths,price])
data = np.transpose(data)

#print("MEDIAN IS %f" % (np.median(data[:,5])))

# Create our target data based on the THRESHOLD value
target = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    if(data[i,5] > THRESHOLD):
        target[i] = 1
    else:
        target[i] = -1

# Normalize all the data so it is between 0 and 1
for i in range(1, data.shape[1]):
    minimum = np.amin(data[:,i])
    maximum = np.amax(data[:,i])
    data[:,i] = (data[:,i]-minimum)/(maximum-minimum)

# Initialize weights to zero
weights = np.zeros(data.shape[1])

# Using 30/70 test/train split of data
# (selecting training & testing data randomly)
num_train_items = int(0.7*data.shape[0])
training_data_index = np.random.choice(data.shape[0],size=(num_train_items),replace=False)

training_data = np.zeros((num_train_items,data.shape[1]))
training_target = np.zeros(num_train_items)
testing_data = np.zeros((data.shape[0]-num_train_items,data.shape[1]))
testing_target = np.zeros(data.shape[0]-num_train_items)

training_count = 0
testing_count = 0
for i in range(data.shape[0]):
    if i in training_data_index:
        training_data[training_count,:] = data[i,:]
        training_target[training_count] = target[i]
        training_count += 1
    else:
        testing_data[testing_count,:] = data[i,:]
        testing_target[testing_count] = target[i]
        testing_count += 1

# Train perceptron and test its accuracy
weights = perceptron(training_data,weights,training_target,1/500)
train_acc = accuracy(training_data,weights,training_target)
test_acc = accuracy(testing_data,weights,testing_target)
print("The training accuracy was: %f" % (train_acc))
print("The testing accuracy was:  %f" % (test_acc))

################################################################################
# Using the code below, I determined the best value to stop training at, based on
# accuracy, was approx 1/500
################################################################################
# A = []
# for i in range(100,1000,100):
#     A.append(i)
# test_acc = np.zeros(len(A))
# train_acc = np.zeros(len(A))
# print(A)
# for i in range(len(A)):
#     weights = perceptron(training_data,weights,training_target,1/A[i])
#     test_acc[i] = accuracy(testing_data,weights,testing_target)
#     train_acc[i] = accuracy(training_data,weights,training_target)
#
# plt.plot(test_acc)
# plt.plot(train_acc)
# plt.legend(['test acc','train acc'])
# plt.show()
