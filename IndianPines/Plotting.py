from IndianPines import IndianPines_Input
import matplotlib.pyplot as plt
from spectral import get_rgb,imshow


input = IndianPines_Input.IndianPines_Input()

plt.figure()
plt.axis("off")
img = get_rgb(input.complete_gt, color_scale=input.color_scale)

imshow(img, fignum=1)