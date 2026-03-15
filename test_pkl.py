import pickle
from pypcd4 import PointCloud


with open('.\\tests\\annotations\\cuboids\\00.pkl.', 'rb') as f:
    data = pickle.load(f)

# print(data.columns.tolist())
print(data.iloc[0])


with open('.\\tests\\lidar\\00.pkl.', 'rb') as f:
    data = pickle.load(f)

# print(data)
# print(data.to_numpy()[:, :4])
pc = PointCloud.from_xyzi_points(data.to_numpy()[:, :4])
pc.save("test.pcd")