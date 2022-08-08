import zipfile
import _pickle as pickle

# with zipfile.ZipFile('dev_uncased.zip') as f:
#     print(f.infolist())
#     data = f.read('dev_uncased.data')
#     data = pickle.loads(data)
#     print(data[:2])

with open(r'train_uncased.data', 'rb') as f:
    data = pickle.load(f)

print(data[0].keys())
