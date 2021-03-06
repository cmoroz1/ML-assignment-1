import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001

def mean_squared_error(data,weights,pred,min_pred,max_pred):
    total = 0
    for i in range(data.shape[0]):
        diff = np.dot(weights,data[i,:]) - pred[i]
        total += (diff)**2
        #scaled_diff = diff*(max_pred-min_pred) + min_pred
        #total += np.abs(scaled_diff)
    return (total/data.shape[0])

# Updates weights using Gradient Descent Algorithm
def update_weights(data, weights, target):
    dw = np.zeros(weights.shape)
    for i in range(dw.shape[0]):
        total = 0
        for d in range(data.shape[0]):
            total += (target[d]-np.dot(weights,data[d,:]))*(-data[d,i])
        dw[i] = total
    return weights - LEARNING_RATE*dw

# Open data file and read in all the data
f = open('housing_prices.txt','r')
lines = f.readlines()
f.close()

# Get rid of all the tabs and new lines inbetween data
for i in range(len(lines)):
    if '\n' in lines[i]:
        lines[i] = lines[i][:-1]
    lines[i] = lines[i].split('\t')

# The labels are the first row and the data is the rest of the file
labels = lines[0]
all_data = lines[1:]

# Initialize empty numpy arrays
sqft = np.zeros(len(all_data))
price = np.zeros(len(all_data))
city = []
bedrooms = np.zeros(len(all_data))
baths = np.zeros(len(all_data))

# Put data into numpy arrays
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
data = np.array([np.ones(sqft.shape[0]),sqft,city,bedrooms,baths])
data = np.transpose(data)
target = np.array(price)

# Normalize all the data so it is between 0 and 1
for i in range(1,data.shape[1]):
    minimum = np.amin(data[:,i])
    maximum = np.amax(data[:,i])
    data[:,i] = (data[:,i]-minimum)/(maximum-minimum)

# Save the min and max target (may be needed later?)
min_target = np.amin(target)
max_target = np.amax(target)
target = (target-min_target)/(max_target-min_target)

# Initialize weights semi-randomly
weights = np.random.rand(data.shape[1])
# weights = np.ones(data.shape[1])

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


# TRAINING PART
n = 20000
# errors_train = np.zeros(n)
# errors_test = np.zeros(n)

for i in range(n):
    weights = update_weights(training_data,weights,training_target)
    # errors_train[i] = mean_squared_error(training_data,weights,training_target,min_target,max_target)
    # errors_test[i] = mean_squared_error(testing_data,weights,testing_target,min_target,max_target)

# plt.plot(errors_train)
# plt.plot(errors_test)
# plt.legend(['training error','testing error'])
# plt.show()

print("The mean squared training error is: %f" % (mean_squared_error(training_data,weights,training_target,min_target,max_target)))
print("The mean squared testing error is:  %f" % (mean_squared_error(testing_data,weights,testing_target,min_target,max_target)))
