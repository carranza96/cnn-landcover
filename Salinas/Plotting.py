from Salinas import Salinas_Input
# from Flevoland import Flevoland_Input
from SanFrancisco import SanFrancisco_Input
from IndianPines import IndianPines_Input
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from spectral import imshow


# input = IndianPines_Input.IndianPines_Input()
# input = Salinas_Input.Salinas_Input()
# input = Flevoland_Input.Flevoland_Input()
input = SanFrancisco_Input.SanFrancisco_Input()
plt.figure()
plt.axis("off")

# imshow(input.input_data, (29,19,9), fignum=1)
# imshow(input.input_data[150:200,150:200],(29,19,9))

raw = envi.open('SanFrancisco/resultados/RFps5.hdr',
                                'SanFrancisco/resultados/RFps5.raw')

imshow(raw, color_scale= input.color_scale)