import pandas as pd

german = pd.read_excel('concatenated_german.xlsx')
parkinson = pd.read_excel('concatenated_parkinson.xlsx')
spam = pd.read_excel('concatenated_spam.xlsx')
biodeg = pd.read_excel('concatenated_biodeg.xlsx')
ozone = pd.read_excel('concatenated_ozone.xlsx')

pd.concat((german, parkinson, ozone, spam, biodeg)).to_excel('concatenated_all.xlsx')