import numpy as np

THRESHOLD = 500000
LEARNING_RATE = 0.01

def update_weights(data, weights, target):
    dw = np.zeros(weights.shape)
    for i in range(dw.shape[0]):
        total = 0
        for d in range(data.shape[0]):
            total += (target[d]-np.dot(weights,data[d,:]))*(data[d,i])
        dw[i] = total
    return weights + LEARNING_RATE*dw

def mean_squared_error(data,weights,pred):
    total = 0
    for i in range(data.shape[0]):
        total += (np.dot(weights,data[i,:]) - pred[i])**2
    return total/data.shape[0]

def perceptron(data, weights, target):
    old_error = mean_squared_error(data,weights,target)
    new_error = 10000
    while(np.abs(old_error-new_error) > 0.00001):
        weights = update_weights(data,weights,target)
        old_error = new_error
        new_error = mean_squared_error(data,weights,target)
    return weights

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

f = open('housing_prices.txt','r')
lines = f.readlines()
f.close()

for i in range(len(lines)):
    if '\n' in lines[i]:
        lines[i] = lines[i][:-1]
    lines[i] = lines[i].split('\t')

labels = lines[0]
all_data = lines[1:]

sqft = np.zeros(len(all_data))
price = np.zeros(len(all_data))
city = []
bedrooms = np.zeros(len(all_data))
baths = np.zeros(len(all_data))

for i in range(len(all_data)):
    row = all_data[i]
    sqft[i] = float(row[0])
    price[i] = float(row[1])
    city.append(row[2])
    bedrooms[i] = float(row[3])
    baths[i] = float(row[4])

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
target = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    if(data[i,5] > THRESHOLD):
        target[i] = 1
    else:
        target[i] = -1

# Normalize all the data so it is between 0 and 1
max_price = np.amax(price)
for i in range(data.shape[1]):
    data[:,i] = data[:,i]/np.amax(data[:,i])

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


weights = perceptron(training_data,weights,training_target)
train_acc = accuracy(training_data,weights,training_target)
test_acc = accuracy(testing_data,weights,testing_target)
print("The training accuracy was: %f" % (train_acc))
print("The testing accuracy was:  %f" % (test_acc))
