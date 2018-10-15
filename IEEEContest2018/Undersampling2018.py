from imblearn.under_sampling import EditedNearestNeighbours
from IEEEContest2018 import Input2018
from collections import Counter

input = Input2018.Input2018()
patch_size = 5
X, y = input.read_train_data(patch_size)


X_reshaped = X.reshape(X.shape[0], patch_size*patch_size*input.bands)[:3000]
y = y[:3000]
print(sorted(Counter(y).items()))



enn = EditedNearestNeighbours(n_jobs=8)
X_resampled, y_resampled = enn.fit_resample(X_reshaped, y)
X_resampled = X_resampled.reshape(X_resampled.shape[0], patch_size, patch_size, input.bands)
print(sorted(Counter(y_resampled).items()))