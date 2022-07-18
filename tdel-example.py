import numpy as np
import pygravlens as gl
from astropy.cosmology import Planck15 as cosmo

# the model we will use
modeltype = 'SIS'
p = [0.0,0.0,1.0]
src = [0.2,0.0]

# specify the lens and source redshifts
zlens = 0.3
zsrc = 2.0
# note that the analysis needs comoving distances
Dl = cosmo.comoving_distance(zlens)
Ds = cosmo.comoving_distance(zsrc)

############################################################
# Approach 1: specify the model with cosmological distances
############################################################

plane1 = gl.lensplane(modeltype,p,Dl=Dl)
lens1 = gl.lensmodel([plane1],Ds=Ds)
lens1.tile()
imgarr1,muarr1,dtarr1 = lens1.findimg(src)
print('Approach 1:')
for i in range(len(imgarr1)):
    print(f'{imgarr1[i,0]:13.6e} {imgarr1[i,1]:13.6e} {muarr1[i]:13.6e} {dtarr1[i]:13.6e}')

############################################################
# Approach 2: specify the model relative to Ds
############################################################

plane2 = gl.lensplane(modeltype,p,Dl=Dl/Ds)
lens2 = gl.lensmodel([plane2])
lens2.tile()
imgarr2,muarr2,dtarr2 = lens2.findimg(src)
print('Approach 2:')
for i in range(len(imgarr2)):
    print(f'{imgarr2[i,0]:13.6e} {imgarr2[i,1]:13.6e} {muarr2[i]:13.6e} {dtarr2[i]:13.6e}')

############################################################
# In approach 2, time delays are dimensionless and in units
# of Ds/c*(arcsec/rad)^2, assuming the image positions are
# in units of arcsec. That scale factor has already been
# computed for lens1, so we can apply it as a cross-check.
############################################################

print('Apply tfac:')
for i in range(len(imgarr2)):
    print(f'{imgarr2[i,0]:13.6e} {imgarr2[i,1]:13.6e} {muarr2[i]:13.6e} {lens1.tfac*dtarr2[i]:13.6e}')

