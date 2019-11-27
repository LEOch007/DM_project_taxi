import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Task 5.1:
# It is Monday 8am, the prime time for every Taxi Driver.
# Daniel have received three orders at the same time.
# Assume Daniel are equally far from these three orders.
# Please help Daniel to pick the one with largest income (fare + tips).
# The fields fare, tips and best are masked in testing.

# get the whole training dataset
train_data = pd.read_csv('training.csv', sep=',', parse_dates=['trip_start_timestamp', 'trip_end_timestamp'])
test_data = pd.read_csv('5_1_testing.csv', sep=',', parse_dates=['trip_start_timestamp'])

# get the traing dataset whose trip start time is 8am
train_data = train_data.set_index('trip_start_timestamp')
train_data_8am = train_data.between_time(start_time='8:00', end_time='8:01')

# data preprocessing
# drop unreasonable values
train_data_8am = train_data_8am.dropna()
train_data_8am = train_data_8am.loc[train_data_8am['trip_seconds'] > 0]
train_data_8am = train_data_8am.loc[train_data_8am['trip_miles'] > 0]
train_data_8am = train_data_8am.loc[train_data_8am['fare'] > 0]
train_data_8am = train_data_8am.loc[train_data_8am['tips'] >= 0]
train_data_8am = train_data_8am.loc[train_data_8am['tolls'] >= 0]
train_data_8am = train_data_8am.loc[train_data_8am['extras'] >= 0]
train_data_8am = train_data_8am.loc[train_data_8am['trip_total'] > 0]

# plot scatter figures, take 'fare' attribute as an example
plt.scatter(range(len(train_data_8am)), train_data_8am['fare'])
plt.yticks(range(0, 1100, 50))
plt.ylabel('fare')
plt.title('fare variable scatter figure to find outliers')
plt.savefig('fare variable scatter figure')
plt.close()

# drop outliers
train_data_8am = train_data_8am.loc[train_data_8am['trip_seconds'] < 10000]
train_data_8am = train_data_8am.loc[train_data_8am['trip_miles'] < 50]
train_data_8am = train_data_8am.loc[train_data_8am['fare'] < 100]
train_data_8am = train_data_8am.loc[train_data_8am['tips'] < 20]
train_data_8am = train_data_8am.loc[train_data_8am['tolls'] < 10]
train_data_8am = train_data_8am.loc[train_data_8am['extras'] < 10]
train_data_8am = train_data_8am.loc[train_data_8am['trip_total'] < 100]
train_data_8am = train_data_8am.reset_index()

# scatter figure of (pickup_latitude, pickup_longitude, fare)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(train_data_8am['pickup_latitude'], train_data_8am['pickup_longitude'], train_data_8am['fare'])
ax.set_zlabel('fare')
ax.set_ylabel('pickup_longitude')
ax.set_xlabel('pickup_latitude')
plt.savefig('pickup_location fare 3-D scatter figure')
plt.close()

# scatter figure of (pickup_latitude, pickup_longitude, tips)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(train_data_8am['pickup_latitude'], train_data_8am['pickup_longitude'], train_data_8am['tips'])
ax.set_zlabel('tips')
ax.set_ylabel('pickup_longitude')
ax.set_xlabel('pickup_latitude')
plt.savefig('pickup_location tips 3-D scatter figure')
plt.close()

# seperate train_data into train_data1(99%) and validation_data(1%)
train1_data_8am = train_data_8am.sample(frac=0.99)
train1_data_8am = train1_data_8am.reset_index()
validation_data_8am = train_data_8am[~train_data_8am.index.isin(train1_data_8am.index)]
validation_data_8am = validation_data_8am.reset_index()

def get_one_test_example(s):
    order_number = [1, 2, 3]
    order_latitude = []
    order_longitude = []
    for i in order_number:
        order_latitude.append(s['order'+str(i)+'_pickup_latitude'])
        order_longitude.append(s['order' + str(i) + '_pickup_longitude'])
    res = pd.DataFrame(columns=['order_number', 'pickup_latitude', 'pick_longitude'])
    res['order_number'] = pd.Series(order_number)
    res['pickup_latitude'] = pd.Series(order_latitude)
    res['pickup_longitude'] = pd.Series(order_longitude)
    return res

plt.scatter(train_data_8am['pickup_latitude'], train_data_8am['pickup_longitude'], color='b', label='train')
test_example = get_one_test_example(test_data.loc[0])
plt.scatter(test_example['pickup_latitude'], test_example['pickup_longitude'], color='r', label='test')
plt.xlabel('pickup_latitude')
plt.ylabel('pickup_longitude')
plt.legend()
plt.savefig('pickup_points & one test example points scatter figure')
plt.close()

def k_nearest_neighbors_predictor(test, k, train_data):
    """
    Finding the test order's nearest k neighbors,
    Using the mean value of them to estimate the corresponding value of test order

    """
    tem_data = train_data
    tem_data['distance'] = ((tem_data['pickup_latitude'] - test['pickup_latitude']) ** 2 \
                             + (tem_data['pickup_longitude'] - test['pickup_longitude']) ** 2).apply(sqrt)

    tem_data.sort_values('distance', inplace=True, ascending=False)
    estimate_fare = round(tem_data['fare'][:(k+1)].mean(), 4)
    estimate_tips = round(tem_data['tips'][:(k+1)].mean(), 4)

    return (estimate_fare, estimate_tips)

# get the MSE value of kNN method using validation data to evaluate the model
validation_position = validation_data_8am.loc[:, ('pickup_latitude', 'pickup_longitude')]
validation_fare = validation_data_8am.loc[:, ('fare')]
validation_tips = validation_data_8am.loc[:, ('tips')]
fare_squred_error_sum = 0
tips_squred_error_sum = 0

for i in range(0,len(validation_position)):
    e_fare, e_tips = k_nearest_neighbors_predictor(validation_position.loc[i], 10, train1_data_8am)
    fare_squred_error_sum += (e_fare - validation_fare[i]) ** 2
    tips_squred_error_sum += (e_tips - validation_tips[i]) ** 2

MSE_fare = sqrt(fare_squred_error_sum / len(validation_position))
MSE_tips = sqrt(tips_squred_error_sum / len(validation_position))

print('fare MSE of validation data:', MSE_fare)
print('tips MSE of validation data:', MSE_tips)

# get the prediction of test data
for i in range(len(test_data)):
    ex = get_one_test_example(test_data.loc[i])
    best_order = 0
    best_profit = 0
    for j in range(len(ex)):
        e_fare, e_tips = k_nearest_neighbors_predictor(ex.loc[j], 10, train_data_8am)
        test_data.loc[i, 'order'+str(j+1)+'_fare'] = e_fare
        test_data.loc[i, 'order'+str(j+1)+'_tips'] = e_tips
        if (e_fare + e_tips) > best_profit:
            best_order = j+1
            best_profit = e_fare + e_tips
    test_data.loc[i, 'best'] = 'order' + str(best_order)

test_data.to_csv('5_1_result.csv', index=0)

