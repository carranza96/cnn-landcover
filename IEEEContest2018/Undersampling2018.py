from imblearn.under_sampling import CondensedNearestNeighbour,NearMiss, OneSidedSelection, \
    NeighbourhoodCleaningRule, RepeatedEditedNearestNeighbours,RandomUnderSampler
from IEEEContest2018 import Input2018
from collections import Counter

input = Input2018.Input2018()
patch_size = 1
X, y = input.read_train_data(patch_size)
print(len(X))

X_reshaped = X.reshape(X.shape[0], patch_size*patch_size*input.bands)
print(sorted(Counter(y).items()))
#
#
#
# enn = RepeatedEditedNearestNeighbours(n_jobs=8)
# enn = CondensedNearestNeighbour(sampling_strategy=[1.7,8],random_state=0)
enn = RandomUnderSampler(sampling_strategy={0:10000,1:15000,3:10000,4:10000,5:10000,7:15000,8:20000,9:10000, 10:10000, 12:10000,13:10000,
                                                    14:10000,15:10000,17:10000,18:10000,19:10000},random_state=0)
X_resampled, y_resampled = enn.fit_resample(X_reshaped, y)

#
# nm = NearMiss(version=3, n_jobs=8)
# X_resampled, y_resampled = nm.fit_resample(X_reshaped, y)
# X_resampled = X_resampled.reshape(X_resampled.shape[0], patch_size, patch_size, input.bands)
print(sorted(Counter(y_resampled).items()))
print(len(X_resampled))


# input.load_image()
# img = input.image.load().transpose((2,0,1))