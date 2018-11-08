from SanFrancisco import SanFrancisco_Input
import matplotlib.pyplot as plt
from spectral import get_rgb,imshow,save_rgb
import spectral.io.envi as envi


input = SanFrancisco_Input.SanFrancisco_Input()

plt.figure()
plt.axis("off")
img = get_rgb(input.train_data, color_scale=input.color_scale)
outmap = envi.open('SanFrancisco/mejor_resultado_sinRot/sanfrancisco_filtro5_5it.hdr',
                                'SanFrancisco/mejor_resultado_sinRot/sanfrancisco_filtro5_5it.raw')
save_rgb("SanFrancisco/dase_sanfrancisco_outmap.png",outmap, color_scale= input.color_scale,format='png')
save_rgb("SanFrancisco/dase_sanfrancisco_train.png",input.train_data, color_scale= input.color_scale,format='png')


imshow(img, fignum=1)