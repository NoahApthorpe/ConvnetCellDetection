f = open('dataset.spec','w')

s = ''
for i in range(1,30) :
    s += '[image' + str(i) +']\n'
    s += 'fnames = /home/znn-release/experiments/two_photon_seg/factor2/jan_new_pp/centroid/stk' + str(i) + '.tif\n'
    s += 'pp_types = standard3D\n'
    s += 'is_auto_crop = yes\n\n'

    s += '[label' + str(i) +']\n'
    s += 'fnames = /home/znn-release/experiments/two_photon_seg/factor2/jan_new_pp/centroid/roi' + str(i) + '.tif\n'
    s += 'pp_types = auto\n'
    s += 'is_auto_crop = yes\n\n'
    
    s += '[sample' + str(i) +']\n'
    s += 'input = ' + str(i) + '\n'
    s += 'output = ' + str(i) + '\n\n'

f.write(s)
f.close()

