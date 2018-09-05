import IndianPines_Input
import numpy as np
from random import shuffle,seed
num_classes = IndianPines_Input.NUM_CLASSES

def train_test_split(X,y,num_folds=1):
    train_indices = []
    test_indices = []
    classes_dict = {c: np.where(y==c)[0] for c in range(num_classes)}
    #class_proportions = {c: len(n)/dataset_size for (c,n) in classes_dict.items()}
    #print(class_proportions)
    random_seeds = [1,2,3,7,11,13,19,23,29,31]

    if num_folds==1:
        folds = [[],[]]
    else:
        folds = [ [[],[]] for _ in range(num_folds)]


    for c in range(num_classes):
        indices = classes_dict[c]
        num_elements = len(indices)

        if num_elements<200:
            train_size = int(round(num_elements*0.75/5)*5)
        elif num_elements<500:
            train_size = 100
        elif num_elements < 1000:
            train_size = 150
        else:
            train_size = 250

        for f in range(num_folds):

            seed(random_seeds[f])
            shuffle(indices)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            if num_folds==1:
                folds[0].extend(train_indices)
                folds[1].extend(test_indices)
            else:
                folds[f][0].extend(train_indices)
                folds[f][1].extend(test_indices)

    return np.asarray(folds)
