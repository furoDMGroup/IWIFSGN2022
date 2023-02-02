import requests
import os.path
import setup_path as pth


datasets = [('biodeg', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'),
            ('german', 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric'),
            ('ozone', 'https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data'),
            ('parkinson', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00301/Parkinson_Multiple_Sound_Recording.rar'),
            ('spam', 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')]

os.chdir(pth.datasets_path)
print('Started downloading data into: ' + pth.datasets_path)
for dataset in datasets:
    if os.path.isfile(dataset[0]):
        pass
    else:
        url = dataset[1]
        r = requests.get(url, allow_redirects=True)
        open(dataset[0], 'wb').write(r.content)


print('Succesfully downloaded data')
print('The parkinson dataset is a rar archive and since it is a proprietary format, some'
      'additional software is usually needed to unpack this archive.')
print('Please unpack this archive by yourself.')
