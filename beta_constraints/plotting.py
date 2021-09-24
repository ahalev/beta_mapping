import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.transforms as tx
from matplotlib import colors

def get_all_boundary_edges(bool_img):
    """
    Get a list of all edges
    (where the value changes from 'True' to 'False') in the 2D image.
    Return the list as indices of the image.
    """
    ij_boundary = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            ij_boundary.append(np.array([[i, j+1],
                                         [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            ij_boundary.append(np.array([[i+1, j],
                                         [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            ij_boundary.append(np.array([[i, j],
                                         [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            ij_boundary.append(np.array([[i, j],
                                         [i, j+1]]))
    if not ij_boundary:
        return np.zeros((0, 2, 2))
    else:
        return np.array(ij_boundary)




def close_loop_boundary_edges(xy_boundary, clean=True):
    """
    Connect all edges defined by 'xy_boundary' to closed
    boundary lines.
    If not all edges are part of one surface return a list of closed
    boundaries is returned (one for every object).
    """

    boundary_loop_list = []
    while xy_boundary.size != 0:
        # Current loop
        xy_cl = [xy_boundary[0, 0], xy_boundary[0, 1]]  # Start with first edge
        xy_boundary = np.delete(xy_boundary, 0, axis=0)

        while xy_boundary.size != 0:
            # Get next boundary edge (edge with common node)
            ij = np.nonzero((xy_boundary == xy_cl[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                xy_cl.append(xy_cl[0])
                break

            xy_cl.append(xy_boundary[i, (j + 1) % 2, :])
            xy_boundary = np.delete(xy_boundary, i, axis=0)

        xy_cl = np.array(xy_cl)

        boundary_loop_list.append(xy_cl)

    return boundary_loop_list

def plot_world_outlines(bool_img, ax=None, extent=None, num_points=100, **kwargs):

    color_2 = [x/255 for x in [255, 127, 14]]  # This is the tableau orange
    color_2.append(0.2)
    cmap = colors.ListedColormap(['white', color_2])
    plt.imshow(bool_img, cmap=cmap, extent=extent, label='C')

    if ax is None:
        ax = plt.gca()

    scale_y = (extent[1]-extent[0])/num_points
    scale_x = (extent[3]-extent[2])/num_points
    translation_y = ((extent[1]+extent[0])-num_points)/2
    translation_x = ((extent[3]+extent[2])-num_points)/2

    ij_boundary = get_all_boundary_edges(bool_img=bool_img.T)
    xy_boundary = ij_boundary - 0.5
    xy_boundary = close_loop_boundary_edges(xy_boundary=xy_boundary)

    trans = tx.Affine2D().translate(translation_x, translation_y).scale(scale_x, -scale_y) + ax.transData
    cl = LineCollection(xy_boundary, **kwargs, colors=['tab:orange'])
    cl.set_transform(trans)

    coll = ax.add_collection(cl)
    coll.set_label('C')
