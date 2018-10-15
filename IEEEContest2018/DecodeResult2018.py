import spectral.io.envi as envi
from IEEEContest2018 import Decoder2018 as Decoder
from IEEEContest2018 import Input2018
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spectral import imshow


input = Input2018.Input2018()
save_path = 'IEEEContest2018/resultados/ps5_4Rot/model-5.ckpt'

config = {}
config['patch_size'] = 5
config['in_channels'] = input.bands
config['num_classes'] = input.num_classes
config['kernel_size'] = 3
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 60
config['train_dropout'] = 0.8
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True
config['seed'] = None
config['log_dir'] = 'IEEEContest2018/resultados/ps5_4Rot'
patch_size = 5
raw = Decoder.decode(input, config, save_path)

# Output image
envi.save_image(config['log_dir'] + "ps" + str(patch_size) + ".hdr",
                raw, dtype='uint8', force=True, interleave='BSQ', ext='raw')

output = Decoder.output_image(input, raw)
# view = imshow(output)
# plt.savefig(config['log_dir'] + 'img/' + str(patch_size) +'.png')


# Image with legend
labelPatches = [patches.Patch(color=input.color_scale.colorTics[x + 1] / 255., label=input.class_names[x]) for x in
                range(input.num_classes)]
fig = plt.figure(2)
lgd = plt.legend(handles=labelPatches, ncol=1, fontsize='small', loc=2, bbox_to_anchor=(1, 1))
imshow(output, fignum=2)
# fig.savefig(config['log_dir'] + 'img/' + str(patch_size) + '_lgd.png',
# bbox_extra_artists=(lgd,), bbox_inches='tight')


# save_rgb('ps'+str(patch_size)+'.png', output, format='png')