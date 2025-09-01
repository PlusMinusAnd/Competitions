import pandas as pd

path = './Service/'
check = pd.read_csv(path +'submission_test.csv')

print(pd.value_counts(check['support_needs']))
# 0    8375
# 2    4358
# 1     492