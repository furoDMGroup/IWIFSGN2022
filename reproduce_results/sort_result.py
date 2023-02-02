from pathlib import PurePath
import pandas as pd
import sys
import os.path

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please pass excel file')
        sys.exit()
    data = pd.read_excel(sys.argv[1])
    sorted = data.sort_values(['missing', 'auc'], ascending=[True, False])
    path = os.path.dirname(sys.argv[1]) + '/sorted_' + os.path.basename(sys.argv[1])
    os_independent_path = str(PurePath(path))
    sorted.to_excel(os_independent_path, index=False)
