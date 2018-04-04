
#general
from __future__ import division

from glob import glob
import numpy as np 
from scipy.misc import imsave, imread
from scipy.io import loadmat

from numpy.lib.stride_tricks import as_strided as ast
from scipy.stats import mode as md
import random, string
import os
from joblib import Parallel, delayed, cpu_count

# =========================================================
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape

# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      outpath = outpath+labels[l]+'/'+outfile
      imsave(outpath, tmp)


direc = '/home/filfy/github/outputs/papers/semantic_seg/seabright/train/'
imdirec = '/home/filfy/github/outputs/papers/semantic_seg/seabright/images/'

files = sorted(glob(direc+'*.mat'))

labels = ['terrain', 'cliff','water','veg','sky','foam','sand', 'anthro','road']
 
thres = .9
tile = 96
outpath = 'autoclassified_gc'+str(tile)
#=======================================================
try:
   os.mkdir(outpath)
except:
   pass

for f in labels: ##classes.keys():
   try:
      os.mkdir(outpath+os.sep+f)
   except:
      pass

for f in files:

   dat = loadmat(f)
   res = dat['class']
   fim = imdirec+f.split(os.sep)[-1].replace('_ares.mat','.jpg')
   print('Generating tiles from dense class map ....')
   Z,ind = sliding_window(imread(fim), (tile,tile,3), (int(tile/2), int(tile/2),3)) 

   C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2))) 

   w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath+os.sep, thres) for k in range(len(Z))) 




