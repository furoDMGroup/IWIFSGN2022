import sys
from statistics import stdev
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from dataset.aggregations import *
import setup_path as pth
from classification.k_neighbours import KNNAlgorithmF
from preprocessing.missing_values import MissingValuesInserterColumnsIndependent
from evaluation.missing_values_crossvalidate import cross_validate_missing_in_test

pd.set_option('display.max_colwidth', 100000000)
pd.set_option('display.max_columns', None)
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)
biodeg = pd.read_csv(pth.concatenate_path_os_independent('biodeg.csv'), header=None, sep=';')

X = biodeg.iloc[:, :-1].to_numpy()
y = biodeg.iloc[:, -1].to_numpy()
y = LabelEncoder().fit_transform(y)


missing = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
dfs = []
ind = 0
ks = [(3, 5, 7, 9, 11, 13, 15, 30)]
aggs = (A1Aggregation(), A2Aggregation(),
        A3Aggregation(), A4Aggregation(p=3),
        A5Aggregation(), A6Aggregation(),
        A7Aggregation(), A8Aggregation(),
        A9Aggregation(), A10Aggregation(),

        A11Aggregation(q=-2, s=-0.5),
        A11Aggregation(q=-0.5, s=2.0),
        A11Aggregation(q=0.5, s=1.0),
        A11Aggregation(q=1.0, s=1.0),
        A11Aggregation(q=1.5, s=3.0),
        A11Aggregation(q=2.0, s=2.0),
        A11Aggregation(q=2.0, s=3.0),
        A11Aggregation(q=3.0, s=4.0),
        A11Aggregation(q=4.0, s=5.0),

        A12Aggregation(q=-1.5),
        A12Aggregation(q=-0.5),
        A12Aggregation(q=1.0),
        A12Aggregation(q=1.5),
        A12Aggregation(q=2.0),

        A14Aggregation(q=-1.5, s=-0.5, p=0.5),
        A14Aggregation(q=-0.5, s=0.5, p=0.5),
        A14Aggregation(q=0.5, s=2.0, p=0.5),
        A14Aggregation(q=1.0, s=2.0, p=1.0),
        A14Aggregation(q=3.0, s=3.0, p=0.2),
        A14Aggregation(q=3.0, s=3.0, p=0.8),

        A15Aggregation(q=1.5, p=1.0),
        A15Aggregation(q=2.0, p=0.0),
        A15Aggregation(q=3.0, p=0.0),
        A15Aggregation(q=3.0, p=0.2),
        A15Aggregation(q=3.0, p=0.5),
        A15Aggregation(q=3.0, p=0.8),
        A15Aggregation(q=3.0, p=1.0),
        A15Aggregation(q=5.0, p=0.5))

for miss in missing:
    for k in ks:
        for agg in aggs:
            skf = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
            binaryF = KNNAlgorithmF(missing_representation=-1, r=10, aggregation=agg, k_neighbours=k, seed=5)
            inserter = MissingValuesInserterColumnsIndependent(columns=range(X.shape[1]), nan_representation=-1,
                                                               percentage=miss, seed=True)
            f_result = cross_validate_missing_in_test(binaryF, X, y,
                                                      scoring='roc_auc', return_estimator=True, cv=skf, inserter=inserter, dataset=None)
            p = agg.p if hasattr(agg, 'p') else ''
            q = agg.q if hasattr(agg, 'q') else ''
            s = agg.s if hasattr(agg, 's') else ''
            w = pd.DataFrame({'algorithm': 'f', 'k': str(k), 'r': f_result['estimator'][0].r,
                              'agg': Aggregation.change_aggregation_to_name(f_result['estimator'][0].aggregation),
                              'p': p, 'q': q,'s': s, 'missing': miss,
                              'auc': np.mean(f_result['test_score']), 'stddev': stdev(f_result['test_score']),
                              }, index=[ind])
            print(w)
            dfs.append(w)
            ind += 1

concatenated = pd.concat(dfs)
concatenated.to_excel('concatenated_biodeg.xlsx')
