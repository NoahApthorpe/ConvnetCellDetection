import numpy as np
from scipy.ndimage.interpolation import zoom
import sys

def coord_polar_to_cart(r, theta, center):
    '''Converts polar coordinates around center to cartesian'''
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


#def coord_cart_to_polar(x, y, center):
#    r = np.sqrt((x-center[0])**2, (y-center[1])**2)
#    theta = np.arctan2((y-center[1]), (x-center[0]))
#    return r, theta


def image_cart_to_polar(image, center, initial_radius, final_radius, phase_width, zoom_factor=1):
    '''Converts an image from cartesian to polar coordinates around center'''
    # ToDo: prevent error if radius goes over edge from center
    image = zoom(image, (zoom_factor, zoom_factor), order=4)
    center = (center[0]*zoom_factor + zoom_factor/2, center[1]*zoom_factor + zoom_factor/2)
    initial_radius = initial_radius * zoom_factor
    final_radius = final_radius * zoom_factor
    
    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, phase_width),
                           np.arange(initial_radius, final_radius))

    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x), np.round(y)
    x, y = x.astype(int), y.astype(int)
    
    if len(image.shape) == 3:
        polar = image[x, y, :]
        polar.reshape((final_radius - initial_radius, phase_width, -1))
    else:
        polar = image[x, y]
        polar.reshape((final_radius - initial_radius, phase_width))

    return polar


def find_edge_2d(polar):
    if len(polar.shape) != 2:
        raise ValueError("argument to find_edge_2d must be 2D")

    values_right_shift = np.pad(polar, ((0,0),(0,1)), 'constant', constant_values=0)[:,1:]
    values_closeright_shift = np.pad(polar, ((1,0),(0,1)), 'constant', constant_values=0)[:-1,1:]
    values_awayright_shift = np.pad(polar, ((0,1),(0,1)), 'constant', constant_values=0)[1:,1:]

    values_move = np.zeros((polar.shape[0], polar.shape[1], 3))
    values_move[:,:,0] = np.add(polar, values_closeright_shift)  # closeright
    values_move[:,:,1] = np.add(polar, values_right_shift)  # right
    values_move[:,:,2] = np.add(polar, values_awayright_shift)  # awayright
    values = np.amax(values_move, axis=2)

    directions = np.argmax(values_move, axis=2)
    directions = np.subtract(directions, 1)
    
    edge = []
    mask = np.zeros(polar.shape)
    r = np.argmax(values[:,0])
    edge.append((r, 0))
    mask[0:r,0] = 1
    for t in range(1,polar.shape[1]):
        r += directions[r,t-1]
        edge.append((r,t))
        mask[0:r,t] = 1

    print values
    print
    print directions
    print
    print np.array(edge)
    print
    print mask
    return np.array(edge), mask


def mask_polar_to_cart(mask, center, phase_width, zoom_factor=1):
    pass


def meanAngle(t1, t2):
    max_t = np.max((t1, t2))
    min_t = np.min((t1, t2))
    if np.abs(t1-t2)) > np.pi:
        max_t -= np.pi*2
    return ((max_t + min_t) / float(2)) % (np.pi*2)

def make_connected_edge(polar_edge, center):
    connected_edge = []
    for i in range(len(polar_edge)):
        j = (i+1) % len(polar_edge)
        pi, pj = polar_edge[i], polar_edge[j]
        ci = coord_polar_to_cart(pi[0], pi[1], center)
        cj = coord_polar_to_cart(pj[0], pj[1], center)
        connected_edge.append(ci)
        
        while np.abs(ci[0]-cj[0]) + np.abs(ci[1]-cj[1]) != 1:
            mean_r = (pi[0] + pj[0]) / float(2)
            mean_theta = meanAngle(pi[1], pj[1])
            mean_x, mean_y = coord_polar_to_cart(mean_r, mean_theta, center)
            need_diag = (mean_x, mean_y) == ci and np.abs(mean_x-cj[0]) + np.abs(mean_y-cj[1]) == 1
            need_diag = need_diag or ((mean_x, mean_y) == cj and np.abs(mean_x-ci[0]) + np.abs(mean_y-ci[1]) == 1)
            if need_diag:
                if mean_theta < np.pi/2:
                    mean_x -= 1
                elif mean_theta < np.pi:
                    mean_y -= 1
                elif mean_theta < np.pi*3/2:
                    mean_x += 1
                else:
                    mean_y += 1
            # TODO ensure mean_x mean_y aren't outside the image
            ci = (mean_x, mean_y)
            connected_edge.append(ci)
        
    return connected_edge


if __name__ == "__main__":
    a = np.arange(49).reshape((7,7))
    b = image_cart_to_polar(a, (3,3), 0, 3, 20, zoom_factor=1)
    #print a
    #print
    #print b

    a = np.array([[0,0,0,0,0,0,0],
         [0,0,0,1,0,0,0],
         [0,0,1,1,1,0,0],
         [0,1,1,1,1,1,0],
         [0,0,1,1,1,0,0],
         [0,0,0,1,0,0,0],
         [0,0,0,0,0,0,0]])
    print a
    print
    b = image_cart_to_polar(a, (3,3), 0, 3, 10, zoom_factor=1)
    print b
    print
    edge, mask = find_edge_2d(b)
    r, theta = edge[:,0], edge[:,1]
    print coord_polar_to_cart(r, theta, (3,3))
