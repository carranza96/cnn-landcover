from Pavia import Pavia_Input
import matplotlib.pyplot as plt
from spectral import get_rgb,imshow


input = Pavia_Input.Pavia_Input()

plt.figure()
plt.axis("off")
img = get_rgb(input.complete_gt, color_scale=input.color_scale)

imshow(img, fignum=1)