import spectral.io.envi as envi
from SanFrancisco import SanFrancisco_Input
from spectral import imshow, get_rgb
from scipy import ndimage, stats
import matplotlib.patches as patches
import matplotlib.pyplot as plt

input = SanFrancisco_Input.SanFrancisco_Input()
img = envi.open('SanFrancisco/mejor_resultado_sinRot/ps5.hdr', 'SanFrancisco/mejor_resultado_sinRot/ps5.raw').load()


def modal(x):
    return stats.mode(x, axis=None)[0][0]

def mode_filter(img):
    return ndimage.generic_filter(img, modal, size=3)


def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)


labelPatches = [patches.Patch(color=input.color_scale.colorTics[x+1]/255., label=input.class_names[x]) for x in range(input.num_classes) ]


view = output_image(input, img)
imshow(view)


print("---------------")
print("Modal filter")
filt_img = img

for n in range(5):
    print("---------------")
    print("Iteration " + str(n))


    filt_img = mode_filter(filt_img)


    view = output_image(input, filt_img)
    fig = plt.figure(n+2)
    lgd = plt.legend(handles=labelPatches, ncol=1, fontsize='small', loc=2, bbox_to_anchor=(1, 1))
    imshow(view, fignum=n+2)
    fig.savefig("SanFrancisco/filt3_lgd", bbox_extra_artists=(lgd,), bbox_inches='tight')

envi.save_image("SanFrancisco/sanfrancisco_filtro3_5it.hdr", filt_img, dtype='uint8', force=True, interleave='BSQ', ext='raw')

