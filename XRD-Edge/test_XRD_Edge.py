# ---------------------------------------------------------
# Example Code for XRD-Edge Generation
# Licensed under The KIST License
# Written by CSRC
# ---------------------------------------------------------
import os
import natsort
import numpy as np
from XRD_Edge import XRD_Edge
from PIL import Image


def call_XRD_Edge(sd, folder, f):
    mp_id = f.split("_")
    filename = str(os.path.join(folder, f))
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype=np.float32)
    img.close()

    lDP = XRD_Edge(data)

    if mp_id[1] == 'x.png':
        result_r = lDP.create_descriptor(1)
        img = Image.fromarray(result_r.astype(np.uint8), 'RGB')
        img.save('Sample Dataset/Sample Result/' + sd + '/x/' + mp_id[0] + '_x.jpg')
    elif mp_id[1] == 'y.png':
        result_r = lDP.create_descriptor(2)
        img = Image.fromarray(result_r.astype(np.uint8), 'RGB')
        img.save('Sample Dataset/Sample Result/' + sd + '/y/' + mp_id[0] + '_y.jpg')
    elif mp_id[1] == 'z.png':
        result_r = lDP.create_descriptor(3)
        img = Image.fromarray(result_r.astype(np.uint8), 'RGB')
        img.save('Sample Dataset/Sample Result/' + sd + '/z/' + mp_id[0] + '_z.jpg')
        print('Sample Dataset/Sample Result/' + sd + '/' + mp_id[0] + ' done!')


d = 'Sample Dataset/Sample XRD'
direct = os.listdir(d)
direct.sort()
sorted(direct)


for sd in direct:
    if os.path.isdir('Sample Dataset/Sample Result/' + sd) is False:
        os.mkdir('Sample Dataset/Sample Result/' + sd)
        os.mkdir('Sample Dataset/Sample Result/' + sd + '/x')
        os.mkdir('Sample Dataset/Sample Result/' + sd + '/y')
        os.mkdir('Sample Dataset/Sample Result/' + sd + '/z')

    folder = os.path.join(d, sd)

    if os.path.isdir(folder):
        directFiles = os.listdir(folder)
        files = natsort.natsorted(directFiles)

        for f in files:
            call_XRD_Edge(sd, folder, f)
