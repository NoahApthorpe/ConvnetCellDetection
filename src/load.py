##################################################################
# 
# Load tif videos and ImageJ ROIs into NumPy arrays
#
# Author: Noah Apthorpe
# 
# Description: reads all .tif video files and ImageJ ROI zip archives 
#   (with same names except for extension, e.g. vid_01.tif, vid_01.zip) 
#   in specified directory to python data structures as follows
#   data = [ (video, ROI) ]
#   video = numpy array [frame #, width pixels, height pixels]
#   ROI = binary numpy array [ROI #, width pixels, height pixels] 
#         where 1-valued pixels are in ROI
#   also returns file_names list s.t. file_names[i] == name of data[i]
#
#  Usage: 
#    import load
#    data = load_data(directory, img_width=N, img_height=M) 
#      # N,M is width and height of video frames in pixels 
#
##################################################################

import numpy as np
from PIL import Image, ImageDraw
import tifffile
import os 
import os.path

# load video and roi files from argument directory
def load_data(directory, img_width, img_height, rois_only=False, no_rois=False):
    assert(not (rois_only and no_rois))
    if directory[-1] != os.path.sep:
        directory += os.path.sep

    file_names = []
    for fn in os.listdir(directory):
        file_base = os.path.basename(fn).rsplit(".")[0]
        file_names.append(file_base)
    file_names = set(file_names)
    if '' in file_names:
        file_names.remove('')
        
    stks = []
    rois = []
    for fn in file_names:
        stack_name = directory+fn+'.tif'
        roi_name = directory+fn+'.zip'
        if rois_only:
            rois.append(load_rois(roi_name, img_width, img_height))
        elif no_rois:
            stks.append(load_stack(stack_name))
        else:
            stks.append(load_stack(stack_name))
            rois.append(load_rois(roi_name, img_width, img_height))
            #data.append((load_stack(stack_name), load_rois(roi_name, img_width, img_height)))
    if rois_only:
        return rois, list(file_names)
    elif no_rois:
        return stks, list(file_names)
    else:
        return stks, rois, list(file_names) 
    #return data, list(file_names)


# tif -> (frame #, width, height)
def load_stack(path):
    with tifffile.TiffFile(path) as im:
        stk = []
        im = im.asarray()
    return np.array(im, dtype='float32')
"""
    im = Image.open(path)
    stk = []
    while True:
        stk.append(np.array(im))
        try:
            im.seek(len(stk))
        except EOFError:
            break
    return np.array(stk, dtype='float32')
"""

# roi zip -> (roi #, width, height)
def load_rois(path, width, height, fill=1, xdisp=0, ydisp=0):
    rois = read_roi_zip(open(path))
    ret = []
    for i,roi in enumerate(rois):
        poly = []
        for x in roi:
            poly.append((x[1] + xdisp,x[0] + ydisp))
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=fill)
        ret.append(np.array(img))
    return np.array(ret,dtype='float32')

###########################################################
# read_roi function 
# https://gist.github.com/luispedro/3437255
# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
# License: MIT
##########################################################

def read_roi(fileobj):
    '''
    points = read_roi(fileobj)

    Read ImageJ's ROI format
    '''
# This is based on:
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html


    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256


    pos = [4]
    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != 'Iout':
        raise IOError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()

    if not (0 <= roi_type < 11):
        raise ValueError('roireader: ROI type %s not supported' % roi_type)

    #if roi_type != 7:
    #    raise ValueError('roireader: ROI type %s not supported (!= 7)' % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat() 
    y1 = getfloat() 
    x2 = getfloat() 
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:,1] = [getc() for i in xrange(n_coordinates)]
    points[:,0] = [getc() for i in xrange(n_coordinates)]
    points[:,1] += left
    points[:,0] += top
    points -= 1
    return points

def read_roi_zip(fname):
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        return [read_roi(zf.open(n))
                    for n in zf.namelist()]
