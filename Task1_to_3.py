import pandas as pd
taxi_data = pd.read_csv('taxi_train.csv',parse_dates=['pickup_datetime'])

import Geohash as gh
taxi_data['pickup_geohash'] = taxi_data['pickup_geohash'].apply(gh.decode)
taxi_data['dropoff_geohash'] = taxi_data['dropoff_geohash'].apply(gh.decode)

taxi_data['pickup_x'] = taxi_data['pickup_geohash'].apply(lambda x: float(x[0]))
taxi_data['pickup_y'] = taxi_data['pickup_geohash'].apply(lambda x: float(x[1]))
taxi_data['dropoff_x'] = taxi_data['dropoff_geohash'].apply(lambda x: float(x[0]))
taxi_data['dropoff_y'] = taxi_data['dropoff_geohash'].apply(lambda x: float(x[1]))

from math import sqrt
taxi_data['distance'] = ((taxi_data['pickup_x'] - taxi_data['dropoff_x']) ** 2 + (taxi_data['pickup_y'] - taxi_data['dropoff_y']) ** 2).apply(sqrt)

removed_rows_num = len(taxi_data.loc[taxi_data['distance'] <= 0])
new_taxi_data = taxi_data.loc[taxi_data['distance'] > 0]
print('The number of rows removed is: %d' % removed_rows_num)

ptime = new_taxi_data.loc[:, ('pickup_datetime', 'passenger')].set_index('pickup_datetime')
num_order_8am_9am = ptime.between_time('08:00', '09:00').shape[0]
num_order_1am_2am = ptime.between_time('01:00', '02:00').shape[0]
print('The number of order between (8am to 9am):', num_order_8am_9am)
print('The number of order between (1am to 2am):', num_order_1am_2am)

time_list = []
for i in range(24):
    for j in ['00', '15', '30', '45']:
        time_str = str(i) + ':' + j
        time_list.append(time_str)

num_list = []    # store the number of orders in corresponding time period

for i in range(len(time_list) - 1):
    num_list.append(ptime.between_time(time_list[i], time_list[i+1], include_start=True,include_end=False).shape[0])
num_list.append(ptime.between_time(time_list[-1], time_list[0], include_start=True,include_end=False).shape[0])

time_order = pd.DataFrame(columns = ['time_period','order_num'])
time_order['time_period'] = pd.Series(time_list)
time_order['order_num'] = pd.Series(num_list)

print(time_order)

from sklearn.cluster import KMeans
import numpy as np

N = new_taxi_data.shape[0]
pickup_data = new_taxi_data.loc[:,('pickup_x','pickup_y')]      # pickup coordinates
dropoff_data = new_taxi_data.loc[:,('dropoff_x','dropoff_y')]   # dropoff coordinates
fitting_data = np.concatenate((pickup_data.values,dropoff_data.values),axis=0)  # horizontal concatenation

location_kmeans = KMeans(n_clusters=30).fit(fitting_data)       # kmeans model
ploc = location_kmeans.labels_[:N].reshape(N,1)                 # pickup location labels
dloc = location_kmeans.labels_[N:].reshape(N,1)                 # dropoff location labels
location_label = np.concatenate((ploc,dloc),axis=1)             # vertical concatenation

num_same_location = sum(location_label[:,0]==location_label[:,1])
rate = num_same_location / N
print('\n')
print('{0:.2f}'.format(rate * 100) + '% order has started from a cluster centers and ends at the same cluster centers')

sample_space = new_taxi_data.index.to_list()
rpickup = np.random.choice(sample_space,100)    # random indexs of pickups
rdropoff = np.random.choice(sample_space,100)   # random indexs of dropoffs

spickup = new_taxi_data.loc[rpickup,('pickup_x','pickup_y')]       # random samples of pickups
sdropoff = new_taxi_data.loc[rdropoff,('dropoff_x','dropoff_y')]   # random samples of dropoffs

# plot the dots
import matplotlib.pyplot as plt
plt.plot(spickup['pickup_x'],spickup['pickup_y'],'bo',label='pickup')        # pickup
plt.plot(sdropoff['dropoff_x'],sdropoff['dropoff_y'],'ro',label='dropoff')   # dropoff
plt.plot(location_kmeans.cluster_centers_[:,0],location_kmeans.cluster_centers_[:,1],'ko',label='cluster centers')   # cluster centers
# plot settings
xmax = max(spickup['pickup_x']) if max(spickup['pickup_x']) >= max(sdropoff['dropoff_x']) else max(sdropoff['dropoff_x'])
xmin = min(spickup['pickup_x']) if min(spickup['pickup_x']) <= min(sdropoff['dropoff_x']) else min(sdropoff['dropoff_x'])
ymax = max(spickup['pickup_y']) if max(spickup['pickup_y']) >= max(sdropoff['dropoff_y']) else max(sdropoff['dropoff_y'])
ymin = min(spickup['pickup_y']) if min(spickup['pickup_y']) <= min(sdropoff['dropoff_y']) else min(sdropoff['dropoff_y'])
eps = 0.02
plt.axis([xmin-eps,xmax+eps,ymin-eps,ymax+eps])
plt.title('The locations of sample taxi records')
plt.xlabel('latitude')
plt.ylabel('longitutde')
plt.legend()
plt.savefig('100 sample training points')
plt.close()

spickup = new_taxi_data.loc[:,('pickup_x','pickup_y')]       # random samples of pickups
sdropoff = new_taxi_data.loc[:,('dropoff_x','dropoff_y')]   # random samples of dropoffs
plt.plot(spickup['pickup_x'],spickup['pickup_y'],'bo',label='pickup')        # pickup
plt.plot(sdropoff['dropoff_x'],sdropoff['dropoff_y'],'ro',label='dropoff')   # dropoff
plt.plot(location_kmeans.cluster_centers_[:,0],location_kmeans.cluster_centers_[:,1],'ko',label='cluster centers')   # cluster centers
# plot settings
xmax = max(spickup['pickup_x']) if max(spickup['pickup_x']) >= max(sdropoff['dropoff_x']) else max(sdropoff['dropoff_x'])
xmin = min(spickup['pickup_x']) if min(spickup['pickup_x']) <= min(sdropoff['dropoff_x']) else min(sdropoff['dropoff_x'])
ymax = max(spickup['pickup_y']) if max(spickup['pickup_y']) >= max(sdropoff['dropoff_y']) else max(sdropoff['dropoff_y'])
ymin = min(spickup['pickup_y']) if min(spickup['pickup_y']) <= min(sdropoff['dropoff_y']) else min(sdropoff['dropoff_y'])
eps = 0.02
plt.axis([xmin-eps,xmax+eps,ymin-eps,ymax+eps])
plt.title('The locations of sample taxi records')
plt.xlabel('latitude')
plt.ylabel('longitutde')
plt.legend()
plt.savefig('all training points')
plt.close()

