import glob
import os
from re import I

from joblib import Parallel, delayed
import tifffile
import numpy as np
from torch import channels_last

vrange = np.arange(256, dtype=np.uint8)
lut_colors = {
    'magenta' : np.stack([vrange, 0*vrange, vrange]),
    'green' : np.stack([0*vrange, vrange, 0*vrange]),
    'yellow' : np.stack([1*vrange, 1*vrange, 0*vrange]),
    'cyan' : np.stack([0*vrange, 1*vrange, 1*vrange]),
    'red' : np.stack([1*vrange, 0*vrange, 0*vrange]),
    'blue' : np.stack([0*vrange, 0*vrange, vrange]),
    'gray' : np.stack([vrange, vrange, vrange]),
}

'''
r04c01f04p01-ch1sk1fk1fl1.tiff'
0123456789012345
          111111
          
r - row
c - column
f - field
p - z slice
sk - time
'''


def parse_name(filename):
    '''
    Parameters
    ----------
    filename : str
        The basename of the file (name without leading path)
        
    Returns
    -------
    welldict : dict 
        the name of the well in c08 format
        c = row 3, 08 column 8
    '''
  
    rowmap = '_abcdefghijklmnop'
    f = filename
    dpos = f.index('-')
    rowstart = f.index('r') + 1
    colstart = f.index('c') + 1
    fieldstart = f.index('f') + 1
    pstart = f.index('p')  + 1
    time_start = f.index('sk') + 2
    channel_start = f.index('ch') + 2
    
    fk_start = f.index('fk')
    row = int(f[rowstart:colstart - 1])
    col = int(f[colstart:fieldstart - 1])
    field = int(f[fieldstart:pstart - 1])
    pz = int(f[pstart:dpos])
    timepoint = int(f[time_start: fk_start])
    channel = int(f[channel_start:time_start - 2])
    
    wellname = f"{rowmap[row]}{col:02d}_{field:04d}.tif"
    
    welldict = dict( 
        filename = filename,
        row = row,
        col = col,
        field = field,
        pz = pz,
        timepoint = timepoint,
        channel = channel,
        wellname = wellname
    )
    
    return welldict
    
    
    
def get_image_files(screen_dir, image_glob='.tiff'):
    image_files = sorted(os.listdir(screen_dir))
    image_files = [x for x in image_files if x.endswith(image_glob)]
    #prefixes = set([x.split('-')[0][:-3] for x in image_files])
    fdict = dict() #{k:[] for k in prefixes} 

    pzmax = 0
    chmax = 0
    tmax = 0
    for f in image_files:
        k = f.split('-')[0][:-3]
        
        welldict = parse_name(f)
        ch = welldict['channel']
        pz = welldict['pz']
        tp = welldict['timepoint'] 
        wellname = welldict['wellname']
        
        if wellname in fdict:
            fdict[wellname].append(welldict)
        else:
            fdict[wellname] = [welldict]
        if ch > chmax:
            chmax = ch

        if pz > pzmax:
            pzmax = pz
            
        if tp > tmax:
            tmax = tp

    return fdict, chmax, pzmax, tmax

def get_image(fdir, imagefiles, nchannels, nz, nt, bin=1):

    stack = None
    for d in imagefiles:
        
        f = d['filename']
        #print(f, d['wellname'])
        pz = d['pz']
        ch = d['channel']
        tp = d['timepoint']
        data = tifffile.imread(f"{fdir}/{f}")
        if bin == 2:
            _dr = data.reshape(
                (data.shape[0]//2, 2, data.shape[1]//2, 2))
            data = _dr.sum(axis=-1).sum(axis=1)
            
        sy, sx = data.shape
        if stack is None:
            stack = np.zeros((nt, nz, nchannels, sy, sx), dtype=data.dtype)
        stack[tp - 1, pz - 1, ch - 1, :, : ] = data

    stack = stack.max(axis=1, keepdims=True)
    #stack = np.expand_dims(stack, 0)
    resdict = {'wellname':d['wellname'],
               'data':stack,
               'axes':'TZCYX'}

    return resdict 

def image_generator(fdir, image_glob='.tiff', bin=1):
    fdict, nchannels, nz = get_image_files(fdir)
    for k, v in fdict.items():
        yield get_image(fdir, k, v, nchannels, nz, bin=bin)
        

def correct_flatness(img, cor_img):
    '''
    Using Sean McKinney's Correct_Flatness
    https://github.com/jouyun/smc-plugins/blob/master/src/main/java/splugins/Correct_Flatness.java
     - DoCorrectUsingProvided
     
     - find the max of the corrected image
     - normalize the correction image by the max
     
     - (original_image - 90)/normalized_correction_image
     
     - using 90 as the background
     
    '''
    
    cmax = cor_img.max()
    n_cor_img = cor_img/cmax
    corrected = (img - 90)/n_cor_img
    
    return corrected

   
def process(s, saveto, resdict, nchannels, cor_img):

    data = resdict['data']
    wellname = resdict['wellname']

    filename = f"{saveto}/{s}" 
    colors = list(lut_colors.keys())[:nchannels]
    luts = [lut_colors[c] for c in colors]
    
    if cor_img is not None:
        data = correct_flatness(data, cor_img)
    

    tifffile.imwrite(filename, data,
            imagej=True,
            photometric='minisblack',
            metadata={'axes': resdict['axes'],
                      'Composite mode':'composite',
                      'LUTs':luts})
    #tifffile.imwrite(filename, data)

def pfunc(d, filenames, **kwargs):
    fdir = kwargs['folder']
    nz = kwargs['nz']
    nt = kwargs['nt']
    nchannels = kwargs['nchannels']
    saveto = kwargs['saveto']
    cor_img = kwargs['correction_image']
    
    image_dict = get_image(fdir, filenames, nchannels, nz, nt, bin=1)
    _ = process(d, saveto, image_dict, nchannels, cor_img)

    return

def run(folders, saveto, globpattern='*.tiff', njobs=2,
        colors=None, correction_image=None):
    
    if isinstance(folders, str):
        folders = [folders]

    if not os.path.exists(saveto):
        try:
            os.makedirs(saveto)
        except:
            print(f"Can't create: {saveto}\n. Exiting")

    print(folders)
    kwargs = {'globpattern':globpattern}
    for i, folder in enumerate(folders):
        print(folder)
        imagefiles, nchannels, nz, nt = get_image_files(folder)

        kwargs['folder'] = folder
        kwargs['nz'] = nz
        kwargs['nt'] = nt
        kwargs['nchannels'] = nchannels
        kwargs['saveto'] = saveto
        kwargs['correction_image'] = correction_image
       
        if njobs < 0: 
            i = 1 
            n = -njobs 
            for k, v in imagefiles.items():
                _ = pfunc(k, v, **kwargs)
                if i >= n:
                    break
                i += 1
        else:
            _ = Parallel(n_jobs=njobs)\
                  (delayed(pfunc)(k, v,  **kwargs) for k, v in imagefiles.items())