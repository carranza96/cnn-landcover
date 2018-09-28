import spectral.io.envi as envi
import numpy as np
from spectral import *
from PIL import Image

# trainingset = envi.open('Avon12/Data/SpecTIR_Set1_Refl.hdr', 'Avon12/Data/SpecTIR_Set1_Refl.img')
# trainingset_gt = envi.open('Flevoland/Data/TrainingSet/cm3253_ground_truth_training.tiff',
#                            'Flevoland/Data/TrainingSet/cm3253_ground_truth_training.tiff')

c = np.fromfile('Flevoland/Data/public/cm3253_c.dat', dtype=float)
l =  np.fromfile('Flevoland/Data/public/cm3253_l.dat', dtype=float)
p = np.fromfile('Flevoland/Data/public/cm3253_p.dat', dtype=float)


gt = np.fromfile('Flevoland/Data/TrainingSet/cm3253_ground_truth_training.tiff', dtype=float)
im = Image.open('Flevoland/Data/TrainingSet/cm3253_ground_truth_training.tiff')
a = np.array(im)