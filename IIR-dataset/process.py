from utils import *


names = get_filenames('raw/', 'csv')

for path in names:
    pd_data = pd.read_csv(path)
    pd_data = pd_data.loc[:, [pd_data.columns[i] for i in [0, 4, 5, 6]]]
    pd_data.columns = ['Concepts', 'Annotator 1', 'Annotator 2', 'Annotator 3']
    # print(pd_data)
    pd_data.to_csv(path.split('/')[1], index=False)
    # break