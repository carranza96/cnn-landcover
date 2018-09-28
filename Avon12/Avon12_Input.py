import spectral.io.envi as envi
from spectral import imshow


trainingset = envi.open('Avon12/Data/SpecTIR_Set1_Refl.hdr', 'Avon12/Data/SpecTIR_Set1_Refl.img')
trainingset_gt = envi.open('Avon12/Data/Green_Block_Reflectance.hdr',
                           'Avon12/Data/Green_Block_Reflectance.spl')