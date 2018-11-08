from Flevoland import Flevoland_Input
import matplotlib.pyplot as plt
from spectral import get_rgb,imshow


input = Flevoland_Input.Flevoland_Input()

plt.figure()
plt.axis("off")
img = get_rgb(input.train_data, color_scale=input.color_scale)

imshow(img, fignum=1)