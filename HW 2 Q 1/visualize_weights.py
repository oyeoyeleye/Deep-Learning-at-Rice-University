from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.cm as cm
import numpy.ma as ma
import matplotlib.pyplot as plt


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


# model = load_model('/home/nickos/servers/storage/py_projects/nk47-assignment2/saved_model.h5')
# model.summary()
# weights = model.get_weights()
#
# w = np.squeeze(weights[0])
# w = np.transpose(w, (2, 0, 1))

# pl.figure(figsize=(15, 15))
# pl.title('conv1 weights')
# nice_imshow(pl.gca(), make_mosaic(w, 6, 6), cmap=cm.binary)
# plt.show()