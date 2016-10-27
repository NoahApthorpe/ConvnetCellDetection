#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
import os
import os.path
from Tkinter import *
from collections import defaultdict
import numpy as np
import ConfigParser

from preprocess import add_pathsep, is_labeled
from load import load_stack, load_rois

class App:

    def __init__(self, root, files, img_width, img_height, islabeled):
        self.current_index = 0
        self.files_keys = files.keys()
        self.files = files
        self.img_width, self.img_height = img_width, img_height
        self.gt_labels = False
        self.manual_thresh = StringVar()
        self.just_set_thresh = False
        self.manual_thresh.trace("w", self.manual_thresh_change)
        self.make_widgets(root, img_width, img_height, islabeled)
        self.load_files()
        
    def make_widgets(self, root, img_width, img_height, islabeled):
        # enclosing frame
        frame = Frame(root)
        frame.grid(padx=10, pady=10)

        # canvas
        self.f = Figure(figsize=(img_width/100.0, img_height/100.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.f, master=frame)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=0, row=1, rowspan=5)

        # other widgets
        self.filename_label = Label(frame, text="Filename")
        self.filename_label.grid(column=0, row=0)
        self.roi_slider = Scale(frame, from_=100, to=0, orient=VERTICAL, length=400, command=self.roi_slider_change)
        self.roi_slider.grid(column=1, row=1, rowspan=4, padx=25)        
        self.image_slider = Scale(frame, from_=0, to=100, orient=HORIZONTAL, length=400, command=self.image_slider_change)
        self.image_slider.grid(column=0, row=6)
        self.cutoff_label = Label(frame, text="set threshold value:")
        self.cutoff_label.grid(column=2, row=1)
        self.cutoff_entry = Entry(frame, textvariable=self.manual_thresh)
        self.cutoff_entry.grid(column=3, row=1)
        self.max_thresh_label = Label(frame, text="current threshold value:")
        self.max_thresh_label.grid(column=2, row=2)
        self.max_thresh_value_label = Label(frame, text="0")
        self.max_thresh_value_label.grid(column=3, row=2)
        self.save_button = Button(frame, text="Save Displayed ROIs", command=self.save_button_press)
        self.save_button.grid(column=2, columnspan=2, row=3)
        prev_next_frame = Frame(frame)
        prev_next_frame.grid(column=2, columnspan=2, row=4)
        self.next_image_button = Button(prev_next_frame, text="Next Image", command=self.next_image_button_press)
        self.next_image_button.grid(column=1, row=0, sticky=W)
        self.prev_image_button = Button(prev_next_frame, text="Previous Image", state=DISABLED, command=self.prev_image_button_press)
        self.prev_image_button.grid(column=0, row=0, sticky=E)
        self.index_label = Label(prev_next_frame, text=self.make_index_label())
        self.index_label.grid(column=0, columnspan=2, row=1)
        if islabeled:
            self.gt_labels_button = Button(frame, text="Show/Hide Ground Truth Labels", command=self.gt_labels_button_press) 
        else:
            self.gt_labels_button = Button(frame, text="Show/Hide Ground Truth Labels", state=DISABLED, command=self.gt_labels_button_press) 
        self.gt_labels_button.grid(column=2, columnspan=2, row=5)

        
    def load_files(self):
        current_files = self.files[self.files_keys[self.current_index]]
        self.filename_label.config(text=self.files_keys[self.current_index])
        self.image = load_stack(current_files[0])
        self.image_slider.config(to=self.image.shape[0]-1)
        self.image_slider.set(0)
        self.image_index = 0
        
        self.convnet_rois = np.load(current_files[1])['arr_0']
        self.convnet_roi_probs = np.load(current_files[1])['arr_1']
        self.indexed_roi_probs = sorted([(v,i) for i,v in enumerate(self.convnet_roi_probs)], reverse=True)
        assert(self.convnet_rois.shape[0] == self.convnet_roi_probs.shape[0])
        self.roi_slider.config(from_=self.convnet_rois.shape[0])

        if self.manual_thresh.get().strip() == "":
            self.roi_slider.set(self.convnet_rois.shape[0])
            self.roi_index = self.convnet_rois.shape[0]
        else:
            new_index = self.convnet_rois.shape[0]
            for j,(p,ind) in enumerate(self.indexed_roi_probs):
                if p < float(self.manual_thresh.get()):
                    new_index = j
                    break
            self.roi_index = new_index
            self.just_set_thresh = True
            self.roi_slider.set(new_index)
                        
        if current_files[2] is not None:
            self.gt_rois = load_rois(current_files[2], self.img_width, self.img_height)
            self.gt_rois = self.gt_rois.max(axis=0)

        self.draw_canvas()


    def draw_canvas(self):        
        self.f.clf()
        overlay = np.zeros((self.image.shape[1], self.image.shape[2], 3))
        overlay[:,:,0] = self.image[self.image_index,:,:]
        overlay[:,:,1] = self.image[self.image_index,:,:]
        overlay[:,:,2] = self.image[self.image_index,:,:]

        current_roi_indices = [i for (v,i) in self.indexed_roi_probs[0:self.roi_index]]
        if len(current_roi_indices) > 0:
            cutoff = self.indexed_roi_probs[self.roi_index-1][0]
            self.max_thresh_value_label.config(text=">= {:.3f}".format(cutoff))
            current_rois_mask = self.convnet_rois[current_roi_indices].max(axis=0)
            overlay[:,:,2][current_rois_mask == 1] = 1    
        else:
            cutoff = self.indexed_roi_probs[0][0]
            self.max_thresh_value_label.config(text="> {:.3f}".format(cutoff))
        
        if self.gt_labels:
            overlay[:,:,0][self.gt_rois == 1] = 1 

        self.f.figimage(overlay)
        self.canvas.draw()

            
    def roi_slider_change(self, value):
        self.roi_index = int(value)
        if self.manual_thresh.get() != "" and not self.just_set_thresh:
            self.manual_thresh.set("")
        self.just_set_thresh = False
        self.draw_canvas()
        
    def image_slider_change(self, value):
        self.image_index = int(value)
        self.draw_canvas()
        
    def save_button_press(self):
        current_roi_indices = [i for (v,i) in self.indexed_roi_probs[0:self.roi_index]]
        current_rois = self.convnet_rois[current_roi_indices]
        current_files = self.files[self.files_keys[self.current_index]]
        new_file = current_files[1][0:-4] + "_MANUAL.npz"
        np.savez_compressed(new_file, rois=current_rois)
        print "saved as " + new_file
        
    def make_index_label(self):
        return str(self.current_index+1) + "/" + str(len(self.files_keys))
        
    def next_image_button_press(self):
        if self.current_index < len(self.files_keys)-1:
             self.current_index += 1
             self.index_label.config(text=self.make_index_label())
             self.load_files()
        if self.current_index == len(self.files_keys)-1:
            self.next_image_button.config(state=DISABLED)
        if self.current_index > 0:
            self.prev_image_button.config(state=NORMAL)
        
    def prev_image_button_press(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.index_label.config(text=self.make_index_label())
            self.load_files()
        if self.current_index == 0:
            self.prev_image_button.config(state=DISABLED)
        if self.current_index < len(self.files_keys)-1:
            self.next_image_button.config(state=NORMAL)

    def gt_labels_button_press(self):
        self.gt_labels = not self.gt_labels
        self.draw_canvas()

    def manual_thresh_change(self, *args):
        if self.manual_thresh.get().strip() != "":
            new_index = self.convnet_rois.shape[0]
            for j,(p,ind) in enumerate(self.indexed_roi_probs):
                if p < float(self.manual_thresh.get()):
                    new_index = j
                    break
            self.roi_index = new_index
            self.just_set_thresh = True
            self.roi_slider.set(new_index)
            self.draw_canvas()    
                            
                                        
def main(main_config_fpath="../main_config_ar.cfg"):
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath,'r'))
    
    # get directory paths
    data_dir = add_pathsep(cfg_parser.get('general', 'data_dir'))
    preprocess_dir = data_dir[0:-1] + "_preprocessed" + os.sep
    postprocess_dir = data_dir[0:-1] + "_postprocessed" + os.sep
    ttv_list = ['training' + os.sep, 'validation' + os.sep, 'test' + os.sep] 
    if not os.path.isdir(data_dir):
        sys.exit("Specified data directory " + data_dir + " does not exist.")
                
    files = defaultdict(lambda: [None, None, None])
    
    for ttv in ttv_list if is_labeled(data_dir) else ['']:
        for f in os.listdir(preprocess_dir + ttv):
            basename, ext = os.path.splitext(f)
            if ext.lower() == '.tif' or ext.lower() == '.tiff':
                if basename[-4:].lower() != "_roi":
                    files[basename][0] = preprocess_dir + ttv + f
        for f in os.listdir(postprocess_dir + ttv):
            basename, ext = os.path.splitext(f)
            if ext == '.npz' and basename[-7:] != "_MANUAL":
                files[basename][1] = postprocess_dir + ttv + f
        for f in os.listdir(data_dir + ttv):
            basename, ext = os.path.splitext(f)
            if ext == '.zip':
                files[basename][2] = data_dir + ttv + f
                
    img_width = cfg_parser.getint('general','img_width')
    img_height = cfg_parser.getint('general', 'img_height')
                       
    root = Tk()
    root.wm_title("ConvnetCellDetection")

    app = App(root, files, img_width, img_height, is_labeled(data_dir))

    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage python " + sys.argv[0] + "config_file_path"
        sys.exit()
    main_config_fpath = sys.argv[1]
    main(main_config_fpath)
    
