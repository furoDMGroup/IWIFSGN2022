import pandas as pd
import sys

if len(sys.argv) != 2:
    raise Exception('Please specify a xlsx results file')

pd.set_option('display.precision', 3)
pd.set_option('display.max_colwidth', 100000000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_excel(sys.argv[1])
global_stats = data[['auc', 'agg']].groupby('agg')\
    .agg({'auc': ['count', max, 'mean', 'median', 'std', sum, 'var']})\
    .sort_values(by=('auc', 'mean'), ascending=False)
print(global_stats)
global_stats.to_excel(sys.argv[1].split('.xlsx')[0] + '_global_summary.xlsx')

stats_missing_levels = data[['auc', 'agg', 'missing']].groupby(['missing', 'agg'])\
    .agg({'auc': [max, 'mean', 'var']}).sort_values(by=[('missing'), ('auc', 'mean')], ascending=[True, False])
print(stats_missing_levels)
stats_missing_levels.to_excel(sys.argv[1].split('.xlsx')[0] + '_missing_levels_summary.xlsx')