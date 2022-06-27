import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.optimize import minimize,fsolve
from shapely.geometry import Point, Polygon
import copy


################################################################################
# UTILITIES
################################################################################

# softening length
soften = 1.0e-6

# some useful matrices
I2 = np.eye(2)
# Pauli matrices used for shear calculations
Pauli_s1 = np.array([[0,1],[1,0]])
Pauli_s3 = np.array([[1,0],[0,-1]])

################################################################################
"""
Generate a 2d grid given a pair of 1d arrays of points in x and y.
The output is structured as follows:
[
  [ [x0,y0],[x0,y1],[x0,y2],... ],
  [ [x1,y0],[x1,y1],[x1,y2],... ],
  ...
]
"""
def mygrid(xarr,yarr):
    return np.moveaxis(np.array(np.meshgrid(xarr,yarr)),0,-1)

################################################################################
"""
Generate random points uniformly on a triangle or set of triangles.
v is a list of triangles with each vertex in a row. Modified from:
https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain
"""
def points_in_triangle(v,n):
    x1 = np.sort(np.random.rand(2,n),axis=0)
    x2 = np.column_stack([x1[0],x1[1]-x1[0],1.0-x1[1]])
    return x2@v

################################################################################
"""
Given a list of vectors, return the ones that are more than tol distance apart.
"""
def get_unique(xraw,tol):
    # seed xnew with the first point
    xnew = np.array([xraw[0]])
    for i in range(1,len(xraw)):
        xtry = xraw[i]
        # compute distance from all previous points
        dist = np.linalg.norm(xnew-xtry,axis=1)
        # if the minimum distance is about the threshold, this point is distinct
        if np.amin(dist)>tol:
            xnew = np.append(xnew,[xtry],axis=0)
    return xnew

################################################################################
"""
Given beta between planes and the distance to one plane, compute the
distance to the other plane. Assumes we are working with distances
that sum (Dij = Dj-Di), and the distances are normalized by Ds.
Recall beta = (Dij*Ds)/(Dj*Dis)
"""
def beta2d(beta,di):
    dj = di/(1.0-beta+beta*di)
    return dj


################################################################################
# MASS MODELS
# TO DO: need to handle divide by zero at centers
################################################################################

################################################################################
"""
point mass
parameters = [x0,y0,thetaE,s]
The softening length s is optional; if not specified, it is set using
the 'soften' global variable.
Softened potential = (1/2)*thetaE^2*ln(s^2+r^2)
"""
def calc_ptmass(parr,x):
    # initialize
    alpha = np.zeros((len(x),2))
    Gamma = np.zeros((len(x),2,2))

    # loop through mass components
    for p in parr:
        # parameters
        x0 = p[0]
        y0 = p[1]
        thetaE = p[2]
        if len(p)>3:
            s = p[3]
        else:
            s = soften

        # positions relative to center
        dx = x - np.array([x0,y0])

        xx = dx[:,0]
        yy = dx[:,1]
        den = s*s+xx*xx+yy*yy
        phix = thetaE**2*xx/den
        phiy = thetaE**2*yy/den
        phixx = thetaE**2*(s*s-xx*xx+yy*yy)/den**2
        phiyy = thetaE**2*(s*s+xx*xx-yy*yy)/den**2
        phixy = -2.0*thetaE**2*xx*yy/den**2

        alpha += np.moveaxis(np.array([phix,phiy]),-1,0)
        Gamma += np.moveaxis(np.array([[phixx,phixy],[phixy,phiyy]]),-1,0)

    return alpha,Gamma

################################################################################
"""
Singular Isothermal Sphere (SIS)
parameters = [x0,y0,thetaE]
"""
def calc_SIS(parr,x):
    # initialize
    alpha = np.zeros((len(x),2))
    Gamma = np.zeros((len(x),2,2))

    # loop through mass components
    for p in parr:
        x0,y0,thetaE = p
        dx = x - np.array([x0,y0])

        r = np.linalg.norm(dx,axis=1)
        cost = dx[:,0]/r
        sint = dx[:,1]/r
        # handle r=0
        indx = np.where(r==0)
        cost[indx] = 1.0
        sint[indx] = 0.0

        phir_r = thetaE/r
        phirr  = 0.0
        phixx  = phir_r*sint*sint + phirr*cost*cost
        phiyy  = phir_r*cost*cost + phirr*sint*sint
        phixy  = (phirr-phir_r)*sint*cost

        alpha += np.array([ phir_r[i]*dx[i] for i in range(len(x)) ])
        Gamma += np.moveaxis(np.array([[phixx,phixy],[phixy,phiyy]]),-1,0)

    return alpha,Gamma

################################################################################
"""
dictionary with known models
"""
massmodel = {
    'ptmass' : calc_ptmass,
    'SIS'    : calc_SIS
}


################################################################################
# LENS PLANE
# Class to handle a single lens plane. The plane can contain an arbitrary
# collection of objects of a single type, along with convergence and shear.
################################################################################

class lensplane:
    
    ##################################################################
    # initialization
    # - ID = string identifying mass model
    # - parr = [[x0,y0,...], [x1,y1,...], ...]
    # - kappa,gammac,gammas = external convergence and shear
    # - Dl_Ds is the ratio (lens distance)/(source distance);
    #   used only in multiplane lensing
    ##################################################################
    
    def __init__(self,ID,parr=[],kappa=0,gammac=0,gammas=0,Dl_Ds=0.5):
        # store the parameters
        self.ID = ID
        self.parr = np.array(parr)
        self.kappa = kappa
        self.gammac = gammac
        self.gammas = gammas
        self.Dl_Ds = Dl_Ds
        if Dl_Ds>=1.0:
            print('Error: lens plane cannot be at or behind source plane')
            return
        # parr should be list of lists; this handles single-component case
        if self.parr.ndim==1: self.parr = np.array([parr])

    ##################################################################
    # compute deflection vector and Gamma tensor at a set of
    # positions; position array can have arbitrary shape
    ##################################################################

    def defmag(self,xarr):
        xarr = np.array(xarr)
        # need special treatment if xarr is a single point
        if xarr.ndim==1:
            oneflag = True
            xarr = np.array([xarr])
        else:
            oneflag = False

        # store shape of xarr so we can apply it to the results arrays
        xarr = np.array(xarr)
        xshape = xarr.shape[:-1]

        # flatten for ease of use here
        xtmp = xarr.reshape((-1,2))

        # call the appropriate lensing function
        atmp,Gtmp = massmodel[self.ID](self.parr,xtmp)

        # factor in convergence and shear
        Mtmp = self.kappa*I2 + self.gammac*Pauli_s3   + self.gammas*Pauli_s1
        atmp += np.array([ Mtmp@x for x in xtmp ])
        Gtmp += np.array([ Mtmp   for x in xtmp ])

        # reshape so the spatial parts match xshape
        alpha_shape = np.concatenate((xshape,[2]))
        Gamma_shape = np.concatenate((xshape,[2,2]))
        alpha = np.reshape(atmp,alpha_shape)
        Gamma = np.reshape(Gtmp,Gamma_shape)

        if oneflag:
            return alpha[0],Gamma[0]
        else:
            return alpha,Gamma

    ##################################################################
    # compute numerical derivatives of alpha and compare them with
    # calculated values of Gamma
    ##################################################################

    def check(self,xarr,h=1.0e-4,floor=1.0e-10):
        # for this purpose, it's fine to have a flattened list of points
        xarr = np.array(xarr).reshape((-1,2))
        # the offsets
        hx = np.array([h,0.0])
        hy = np.array([0.0,h])
        # compute
        a0,G0 = self.defmag(xarr)
        ax,Gx = self.defmag(xarr+hx)
        ay,Gy = self.defmag(xarr+hy)
        # compute numerical 2nd derivatives
        axx = (ax[:,0]-a0[:,0])/h
        axy = (ay[:,0]-a0[:,0])/h
        ayx = (ax[:,1]-a0[:,1])/h
        ayy = (ay[:,1]-a0[:,1])/h
        # construct numerical 2nd deriv matrix
        Gtmp = np.array([[axx,axy],[ayx,ayy]])
        Gtry = np.moveaxis(Gtmp,[0,1],[-2,-1])
        # compare to computed Gamma matrix
        dG = Gtry-G0
        # plot histogram
        tmp1 = np.absolute(dG).flatten()
        tmp2 = np.log10(tmp1[tmp1>floor])
        plt.figure()
        plt.hist(tmp2)
        plt.xlabel('log10(difference in bending angle)')
        plt.ylabel('number')
        plt.show()


################################################################################
# LENS MODEL
# TO DO: add notes
################################################################################

class lensmodel:

    ##################################################################
    # initialize with a list of lensplane structures
    # - xtol is the tolerance used for finding images
    # - position_mode is important for multiplane lensing;
    #   + 'obs' indicates that the specified positions are observed,
    #     so the intrinsic positions must account for foreground bending
    #   + 'fix' indicates that the specified positions are fixed in space
    # - multi_mode specifies how the multiplane lensing weight factors
    #   beta and epsilon are handled:
    #   + [] indicates to compute them based on planes' distances
    #   + [beta,epsilon] indicates that the arrays are passed in
    # - Ddecimals is the number of decimals to use when rounding distances
    ##################################################################

    def __init__(self,plane_list,xtol=1.0e-5,position_mode='obs',multi_mode=[],Ddecimals=3):
        self.xtol = xtol
        self.position_mode = position_mode

        # structures for critical curves and caustics
        self.crit = []
        self.caus = []

        # structures for grid
        rlo = 1.0e-6
        rhi = 2.5
        n0 = 20
        self.maingrid(-rhi,rhi,n0,-rhi,rhi,n0)
        self.galgrid(rlo,rhi,n0,n0)

        # various flags
        self.griddone = False
        self.critdone = False

        # group planes into slabs at the same distance;
        # recall that we round distances using Ddecimals
        dtmp = np.array([ plane.Dl_Ds for plane in plane_list ])
        darr,iarr = np.unique(dtmp.round(Ddecimals),return_inverse=True)
        self.slab_list = []
        for i in range(len(darr)):
            # this slab contains all planes at this distance
            slab = [ plane_list[j] for j in np.where(iarr==i)[0] ]
            self.slab_list.append(slab)
        self.nslab = len(self.slab_list)

        # process multi_mode
        if len(multi_mode)==0:
            # compute beta factors; first append source distance to darr
            darr = np.append(darr,1.0)
            self.beta = np.zeros(self.nslab)
            for j in range(self.nslab):
                self.beta[j] = (darr[j+1]-darr[j])*darr[-1]/(darr[j+1]*(darr[-1]-darr[j]))
            # compute epsilon factors; here it helps to prepend darr with 0,
            # but then we have to take care with the indexing
            self.epsilon = np.zeros(self.nslab)
            darr = np.insert(darr,0,0.0)
            for j in range(1,self.nslab+1):
                self.epsilon[j-1] = (darr[j-1]*(darr[j+1]-darr[j]))/(darr[j+1]*(darr[j]-darr[j-1]))
        elif len(multi_mode)==2:
            beta,epsilon = multi_mode
            if np.isscalar(beta):
                self.beta = np.full(self.nslab,beta)
            else:
                self.beta = np.array(beta)
            if np.isscalar(epsilon):
                self.epsilon = np.full(self.nslab,epsilon)
            else:
                self.epsilon = np.array(epsilon)
            if len(self.beta)!=self.nslab:
                print('Error: incorrect length of beta specified in multi_mode')
                return
            if len(self.epsilon)!=self.nslab:
                print('Error: incorrect length of epsilon specified in multi_mode')
                return
        else:
            print('Error: cannot parse multi_mode argument')
            return

        # see if model is 3d
        self.flag3d = (self.nslab>=2)

        # process pobs and pfix
        if self.flag3d==False:
            # not 3d, so things are simple
            for j in range(self.nslab):
                for plane in self.slab_list[j]:
                    plane.pobs = plane.parr[:,0:2] + 0.0
                    plane.pfix = plane.parr[:,0:2] + 0.0
        elif self.position_mode=='obs':
            # map observed positions back to find intrinsic positions
            for j in range(self.nslab):
                for plane in self.slab_list[j]:
                    plane.pobs = plane.parr[:,0:2] + 0.0
                    plane.pfix,A = self.lenseqn(plane.pobs,stopslab=j)
                    plane.parr[:,0:2] = plane.pfix + 0.0
        # note: 3d with position_mode=='fix' is handled in find_centers()

    ##################################################################
    # report some key information about the model
    ##################################################################

    def info(self):
        print('number of planes:',self.nslab)
        print('maingrid:',self.maingrid_info)
        print('galgrid:',self.galgrid_info)
        if self.flag3d:
            print('model is 3d')
            print('position mode:',self.position_mode)
            print('beta:',self.beta)
            print('epsilon:',self.epsilon)

    ##################################################################
    # lens equation; take an arbitrary set of image positions and return
    # the corresponding set of source positions; can handle multiplane
    # lensing; stopslab can be used to stop at some specified plane,
    # and stopslab<0 means go all the way to the source
    ##################################################################

    def lenseqn(self,xarr,stopslab=-1):
        if stopslab<0: stopslab = len(self.slab_list)
        xarr = np.array(xarr)
        # need special treatment if xarr is a single point
        if xarr.ndim==1:
            oneflag = True
            xarr = np.array([xarr])
        else:
            oneflag = False

        # structures to store everything (all slabs)
        xshape = list(xarr.shape[:-1])
        xall = np.zeros([self.nslab+1]+xshape+[2])
        Aall = np.zeros([self.nslab+1]+xshape+[2,2])
        alphaall = np.zeros([self.nslab+1]+xshape+[2])
        GammAall = np.zeros([self.nslab+1]+xshape+[2,2])

        # set of identity matrices for all positions
        tmp0 = np.zeros(xshape)
        tmp1 = tmp0 + 1.0
        bigI = np.moveaxis(np.array([[tmp1,tmp0],[tmp0,tmp1]]),[0,1],[-2,-1])

        # initialize first slab
        xall[0] = xarr
        Aall[0] = bigI

        # construct the z and A lists by iterating
        for j in range(stopslab):
            # compute this slab
            alpha_now = np.zeros(xshape+[2])
            Gamma_now = np.zeros(xshape+[2,2])
            for plane in self.slab_list[j]:
                alpha_tmp,Gamma_tmp = plane.defmag(xall[j])
                alpha_now += alpha_tmp
                Gamma_now += Gamma_tmp
            # we need Gamma@A, not Gamma by itself
            Gamma_A_now = Gamma_now@Aall[j]
            # store this slab
            alphaall[j] = alpha_now
            GammAall[j] = Gamma_A_now
            # compute the lens equation
            xall[j+1] = xall[j] - self.beta[j]*alphaall[j]
            Aall[j+1] = Aall[j] - self.beta[j]*GammAall[j]
            if j>=1:
                xall[j+1] += self.epsilon[j]*(xall[j]-xall[j-1])
                Aall[j+1] += self.epsilon[j]*(Aall[j]-Aall[j-1])

        # return the desired plane
        if oneflag:
            return xall[stopslab][0],Aall[stopslab][0]
        else:
            return xall[stopslab],Aall[stopslab]

    ##################################################################
    # compute numerical derivatives d(src)/d(img) and compare them
    # with calculated values of inverse magnification tensor
    ##################################################################

    def check(self,xarr,h=1.0e-4,floor=1.0e-10):
        # for this purpose, it's fine to have a flattened list of points
        xarr = np.array(xarr).reshape((-1,2))
        # the offsets
        hx = np.array([h,0.0])
        hy = np.array([0.0,h])
        # compute
        x0,A0 = self.lenseqn(xarr)
        xx,Ax = self.lenseqn(xarr+hx)
        xy,Ay = self.lenseqn(xarr+hy)
        # compute numerical 2nd derivatives
        axx = (xx[:,0]-x0[:,0])/h
        axy = (xy[:,0]-x0[:,0])/h
        ayx = (xx[:,1]-x0[:,1])/h
        ayy = (xy[:,1]-x0[:,1])/h
        # construct numerical 2nd deriv matrix
        Atmp = np.array([[axx,axy],[ayx,ayy]])
        Atry = np.moveaxis(Atmp,[0,1],[-2,-1])
        # compare to computed Gamma matrix
        dA = Atry-A0
        # plot histogram
        tmp1 = np.absolute(dA).flatten()
        tmp2 = np.log10(tmp1[tmp1>floor])
        plt.figure()
        plt.hist(tmp2)
        plt.xlabel('log10(difference in source position)')
        plt.ylabel('number')
        plt.show()

    ##################################################################
    # commands to specify the grid:
    # - maingrid is Cartesian
    # - galgrid is polar grid(s) centered on the mass component(s)
    ##################################################################

    def maingrid(self,xlo,xhi,nx,ylo,yhi,ny):
        self.maingrid_info = [[xlo,xhi,nx],[ylo,yhi,ny]]

    def galgrid(self,rlo,rhi,nr,ntheta):
        self.galgrid_info = [rlo,rhi,nr,ntheta]

    ##################################################################
    # compute the tiling; this is a wrapper meant to be called by user
    ##################################################################

    def tile(self,addlevels=2,addpoints=5,holes=0):
        # find the centers
        self.find_centers()
        # do the (final) tiling
        self.do_tile(addlevels=addlevels,addpoints=addpoints,holes=holes)
        # for plotting the grid
        self.plotimg = collections.LineCollection(self.imgpts[self.edges],color='lightgray')
        self.plotsrc = collections.LineCollection(self.srcpts[self.edges],color='lightgray')

    ##################################################################
    # internal: find the center(s) of the mass component(s)
    ##################################################################

    def find_centers(self):
        centers = []
        
        if self.flag3d==False:

            # model is not 3d, so we can just collect all of the centers
            for j in range(self.nslab):
                for plane in self.slab_list[j]:
                    for p in plane.pobs: centers.append(p)
            self.centers = np.array(centers)

        else:

            # model is 3d, so we need to take care with the centers
            if self.position_mode=='obs':
                # we already processed pobs and pfix in __init__()
                for j in range(self.nslab):
                    for plane in self.slab_list[j]:
                        for p in plane.pobs: centers.append(p)
                self.centers = np.array(centers)
            elif self.position_mode=='fix':
                # the specified positions are fixed, so we need
                # to solve the lens equation (for the appropriate
                # source plane) to find the corresponding observed
                # positions
                for j in range(self.nslab):
                    for plane in self.slab_list[j]:
                        plane.pfix = plane.parr[:,0:2] + 0.0
                        if j==0:
                            # for first slab, just use pfix
                            for p in plane.pfix: centers.append(p)
                        else:
                            # we need to tile with what we have so far
                            self.do_tile(stopslab=j)
                            # solve (intermediate) lens equation to find
                            # observed position(s) of center(s)
                            for pfix in plane.pfix:
                                pobs,mu = self.findimg(pfix,plane=j)
                                for p in pobs: centers.append(p)
                        self.centers = np.array(centers)
            else:
                print('Error: unknown position_mode')
                return

    ##################################################################
    # internal: this is the workhorse that does the tiling
    ##################################################################

    def do_tile(self,stopslab=-1,addlevels=2,addpoints=5,holes=0):

        # construct maingrid
        if len(self.maingrid_info)>0:
            xlo,xhi,nx = self.maingrid_info[0]
            ylo,yhi,ny = self.maingrid_info[1]
            xtmp = np.linspace(xlo,xhi,nx)
            ytmp = np.linspace(ylo,yhi,ny)
            self.maingrid_pts = np.reshape(mygrid(xtmp,ytmp),(-1,2))

        # construct galgrid
        if len(self.galgrid_info)>0:
            rlo,rhi,nr,ntheta = self.galgrid_info
            if nr>0:
                rarr = np.linspace(rlo,rhi,nr)
            else:
                rarr = np.logspace(np.log10(rlo),np.log10(rhi),-nr)
            # set up the basic polar grid
            tarr = np.linspace(0.0,2.0*np.pi,ntheta)
            rtmp,ttmp = np.meshgrid(rarr,tarr[:-1])    # note that we skip theta=2*pi to avoid duplication
            rtmp = rtmp.flatten()
            ttmp = ttmp.flatten()
            xtmp = rtmp[:,np.newaxis]*np.column_stack((np.cos(ttmp.flatten()),np.sin(ttmp.flatten())))
            # place the polar grid at each center
            self.galgrid_pts = []
            for x0 in self.centers:
                self.galgrid_pts.append(x0+xtmp)
            # reshape so it's just a list of points
            self.galgrid_pts = np.reshape(np.array(self.galgrid_pts),(-1,2))

        # positions in image plane, from maingrid and galgrid
        # depending on what is available
        if len(self.maingrid_pts)>0 and len(self.galgrid_pts)>0:
            self.imgpts = np.concatenate((self.maingrid_pts,self.galgrid_pts),axis=0)
        elif len(self.maingrid_pts)>0:
            self.imgpts = self.maingrid_pts
        else:
            self.imgpts = self.galgrid_pts

        # if desired, cut holes in grid around centers
        if holes>0:
            for x0 in self.centers:
                r = np.linalg.norm(self.imgpts-x0,axis=1)
                indx = np.where(r<holes)[0]
                self.imgpts = np.delete(self.imgpts,indx,axis=0)

        # positions in source plane, and inverse magnifications
        u,A = self.lenseqn(self.imgpts,stopslab=stopslab)
        self.srcpts = u
        self.minv = np.linalg.det(A)

        # run the initial triangulation
        self.triangulate()
        # if desired, add points near critical curves
        if addlevels>0 and addpoints>0:
            for ilev in range(addlevels):
                self.addpoints(addpoints)

        # update status
        self.griddone = True

    ##################################################################
    # internal: compute the Delaunay triangulation
    ##################################################################

    def triangulate(self):
        # Delaunay triangulation
        self.tri = Delaunay(self.imgpts)
        self.ntri = len(self.tri.simplices)
        # set up useful arrays:
        # edges = list of all triangle edges
        # *poly = triangles in numpy array
        # *Polygon = triangles in shapely Polygon structure
        # *range = range for each triangle
        self.edges = []
        self.imgpoly = []
        self.srcpoly = []
        self.imgPolygon = []
        self.srcPolygon = []
        self.imgrange = []
        self.srcrange = []
        for simp in self.tri.simplices:
            self.edges.append(sorted([simp[0],simp[1]]))
            self.edges.append(sorted([simp[1],simp[2]]))
            self.edges.append(sorted([simp[2],simp[0]]))
            tmp = np.append(simp,simp[0])
            imgtmp = self.imgpts[tmp]
            srctmp = self.srcpts[tmp]
            self.imgpoly.append(imgtmp)
            self.srcpoly.append(srctmp)
            self.imgPolygon.append(Polygon(imgtmp))
            self.srcPolygon.append(Polygon(srctmp))
            self.imgrange.append([np.amin(imgtmp,axis=0),np.amax(imgtmp,axis=0)])
            self.srcrange.append([np.amin(srctmp,axis=0),np.amax(srctmp,axis=0)])
        self.edges = np.unique(np.array(self.edges),axis=0)    # remove duplicate edges
        self.imgpoly = np.array(self.imgpoly)
        self.srcpoly = np.array(self.srcpoly)
        self.imgrange = np.array(self.imgrange)
        self.srcrange = np.array(self.srcrange)

    ##################################################################
    # internal: add more grid points near critical curves
    ##################################################################

    def addpoints(self,nnew):
        # examine products of magnifications on triangle edges
        magprod0 = self.minv[self.tri.simplices[:,0]]*self.minv[self.tri.simplices[:,1]]
        magprod1 = self.minv[self.tri.simplices[:,1]]*self.minv[self.tri.simplices[:,2]]
        magprod2 = self.minv[self.tri.simplices[:,2]]*self.minv[self.tri.simplices[:,0]]
        # find triangles where magnification changes sign
        tmp = np.zeros(self.ntri)
        tmp[magprod0<0.0] += 1
        tmp[magprod1<0.0] += 1
        tmp[magprod2<0.0] += 1
        crittri = self.imgpts[self.tri.simplices[tmp>0]]
        # pick random points in these triangles
        newimg = points_in_triangle(crittri,nnew).reshape((-1,2))
        u,A = self.lenseqn(newimg)
        newsrc = u
        newminv = np.linalg.det(A)
        # add them to the lists
        self.imgpts = np.append(self.imgpts,newimg,axis=0)
        self.srcpts = np.append(self.srcpts,newsrc,axis=0)
        self.minv = np.append(self.minv,newminv,axis=0)
        # recompute the triangulation
        self.triangulate()

    ##################################################################
    # internal: find triangles that contain a specified source point
    ##################################################################

    def findtri(self,u):
        if self.griddone==False:
            print('Error: tiling has not been completed')
            return []
        uPoint = Point(u)
        # first find all triangles that have x in range
        flags = [ (u[0]>=self.srcrange[:,0,0]) & (u[0]<self.srcrange[:,1,0])
                & (u[1]>=self.srcrange[:,0,1]) & (u[1]<self.srcrange[:,1,1]) ]
        indx = np.arange(self.ntri)
        # check those triangles to see which actually contain the point
        goodlist = []
        for itri in indx[tuple(flags)]:
            # note that we buffer the polygon by an amount given by xtol
            if self.srcPolygon[itri].buffer(3.0*self.xtol).contains(uPoint): goodlist.append(itri)
        # return the list of triangle indices
        return goodlist

    ##################################################################
    # solve the lens equation and report the image(s) for a given
    # source position or set of source positions
    ##################################################################f

    def findimg_func(self,x,u,plane):
        utry,Atry = self.lenseqn(x,plane)
        diff = utry - u
        return diff@diff

    def findimg(self,u,plane=-1):
        if self.griddone==False:
            print('Error: tiling has not been completed')
            return [],[]

        srcarr = np.array(u)
        if srcarr.ndim==1:
            srcarr = np.array([u])
            oneflag = True
        else:
            oneflag = False

        # loop over sources
        imgall = []
        muall = []
        for u in srcarr:
            # find triangles that contain u, and check each of them
            trilist = self.findtri(u)
            imgraw = []
            for itri in trilist:
                # run scipy.optimize.minimize starting from the triangle mean
                tri = self.imgpoly[itri,:3]
                xtri = np.mean(tri,axis=0)
                ans = minimize(self.findimg_func,xtri,args=(u,plane),method='Nelder-Mead',options={'initial_simplex':tri,'xatol':0.01*self.xtol,'fatol':1.e-6*self.xtol**2})
                if ans.success: imgraw.append(ans.x)
            imgraw = np.array(imgraw)
            # there may be duplicate solutions, so extract the unique ones
            imgarr = get_unique(imgraw,self.xtol)
            # compute magnifications
            u,A = self.lenseqn(imgarr)
            muarr = 1.0/np.linalg.det(A)
            # add  to lists
            imgall.append(imgarr)
            muall.append(muarr)

        if oneflag:
            return imgall[0],muall[0]
        else:
            return imgall,muall

    ##################################################################
    # solve the lens equation and report the total magnification for
    # a given source position or set of source positions; this is
    # largely a wrapper for findimg()
    ##################################################################f

    def totmag(self,u,plane=-1):
        imgarr,muarr = self.findimg(u,plane=plane)
        srcarr = np.array(u)
        if srcarr.ndim==1:
            return np.sum(np.absolute(muarr))
        else:
            return [np.sum(np.absolute(mu)) for mu in muarr]

    ##################################################################
    # given an image position, find the corresponding source and then
    # solve the lens equation to find all of the counter images
    ##################################################################

    def findsrc(self,xarr,plane=-1):
        xarr = np.array(xarr)
        if xarr.ndim==1:
            xarr = np.array([xarr])
            oneflag = True
        else:
            oneflag = False

        # loop over sources
        imgall = []
        muall = []
        for x in xarr:
            u,A = self.lenseqn(x,plane)
            imgarr,muarr = self.findimg(u,plane)
            imgall.append(imgarr)
            muall.append(muarr)

        if oneflag:
            return imgall[0],muall[0]
        else:
            return imgall,muall

    ##################################################################
    # compute images of extended source(s)
    # - srcmode = 'disk', 'gaus'
    # - srcarr = list of [u0,v0,radius,intensity]
    # - extent = [ [xlo,xhi,nx], [ylo,yhi,ny ]
    # Returns:
    # - srcmap,imgmap
    ##################################################################

    def extendedimg(self,srcmode='',srcarr=[],extent=[]):
        if len(srcmode)==0:
            print('Error in extendedimg(): srcmode is not specified')
        if len(srcarr)==0:
            print('Error in extendedimg(): srcarr is empty')
        if len(extent)==0:
            print('Error in extendedimg(): extent is empty')

        # want srcarr to be 2d in general
        srcarr = np.array(srcarr)
        if srcarr.ndim==1:
            srcarr = np.array([srcarr])

        # construct the grid
        xlo,xhi,nx = extent[0]
        ylo,yhi,ny = extent[1]
        xtmp = np.linspace(xlo,xhi,nx)
        ytmp = np.linspace(ylo,yhi,ny)
        xarr = mygrid(xtmp,ytmp)
        # map to source plane
        uarr,Aarr = self.lenseqn(xarr)
        # initialize the maps
        srcmap = 0.0*xarr[:,:,0]
        imgmap = 0.0*xarr[:,:,0]

        # loop over sources
        for src in srcarr:
            u0 = src[:2]
            R0,I0 = src[2:4]
            dsrc = np.linalg.norm(xarr-u0,axis=-1)
            dimg = np.linalg.norm(uarr-u0,axis=-1)
            if srcmode=='disk':
                srcflag = np.where(dsrc<=R0)
                imgflag = np.where(dimg<=R0)
                srcmap[srcflag] += I0
                imgmap[imgflag] += I0
            elif srcmode=='gaus':
                srcmap += np.exp(-0.5*dsrc**2/R0**2)
                imgmap += np.exp(-0.5*dimg**2/R0**2)

        # done
        return srcmap,imgmap

    ##################################################################
    # plot magnification map; the range is taken from maingrid,
    # while steps gives the number of pixels in each direction
    ##################################################################

    def plotmag(self,steps=500,signed=True,mumin=-5,mumax=5,title='',file=''):
        # set up the grid
        xlo,xhi,nx = self.maingrid_info[0]
        ylo,yhi,ny = self.maingrid_info[1]
        xtmp = np.linspace(xlo,xhi,steps)
        ytmp = np.linspace(ylo,yhi,steps)
        xarr = mygrid(xtmp,ytmp)
        # compute magnifications
        u,A = self.lenseqn(xarr)
        mu = 1.0/np.linalg.det(A)
        if signed==False:
            mu = np.absolute(mu)
            if mumin<0.0: mumin = 0.0
        # plot
        plt.figure(figsize=(6,6))
        plt.imshow(mu,origin='lower',interpolation='nearest',extent=[xlo,xhi,ylo,yhi],vmin=mumin,vmax=mumax)
        plt.colorbar()
        plt.gca().set_aspect('equal')
        if len(title)>0:
            plt.title('magnification - '+title)
        else:
            plt.title('magnification')
        if len(file)==0:
            plt.show()
        else:
            plt.savefig(file,bbox_inches='tight')

    ##################################################################
    # plot critical curve(s) and caustic(s); different modes:
    #
    # mode=='grid' : the search range is taken from maingrid, with
    # the specified number of steps in each direction
    #
    # mode=='tile1' : points are interpolated from the tiling
    #
    # mode=='tile2' : initial guesses from the tiling are refined
    # using root finding
    ##################################################################

    def plotcrit(self,mode='grid',steps=500,pointtype='line',show=True,title='',file=''):
        self.critdone = False

        if mode=='grid':

            # set up the grid
            xlo,xhi,nx = self.maingrid_info[0]
            ylo,yhi,ny = self.maingrid_info[1]
            xtmp = np.linspace(xlo,xhi,steps)
            ytmp = np.linspace(ylo,yhi,steps)
            xarr = mygrid(xtmp,ytmp)
            # we need the pixel scale(s)
            xpix = xtmp[1]-xtmp[0]
            ypix = ytmp[1]-ytmp[0]
            pixscale = np.array([xpix,ypix])
            # compute magnifications
            u,A = self.lenseqn(xarr)
            muinv = np.linalg.det(A)

            # get the contours where muinv=0 (python magic!)
            plt.figure()
            cnt = plt.contour(muinv,[0])
            plt.title('dummy plot')
            plt.close()

            # initialize lists
            self.crit = []
            self.caus = []

            # loop over all "segments" in the contour plot
            x0 = np.array([xlo,ylo])
            for v in cnt.allsegs[0]:
                # convert from pixel units to arcsec in image plane
                xcrit = x0 + pixscale*v
                ucaus,A = self.lenseqn(xcrit)
                self.crit.append(xcrit)
                self.caus.append(ucaus)

        elif mode=='tile1':

            if self.griddone==False:
                print("Error: plotcrit with mode=='tile1' requires tiling to be complete")
                return
            # get endpoints of all segments
            tmp = self.imgpts[self.edges]
            xA = tmp[:,0]
            xB = tmp[:,1]
            tmp = self.minv[self.edges]
            mA = tmp[:,0]
            mB = tmp[:,1]
            # find segments where magnification changes sign
            indx = np.where(mA*mB<0)[0]
            xA = xA[indx]
            xB = xB[indx]
            mA = mA[indx]
            mB = mB[indx]
            # interpolate
            wcrit = (0.0-mA)/(mB-mA)
            xcrit = (1.0-wcrit[:,None])*xA + wcrit[:,None]*xB
            ucaus,A = self.lenseqn(xcrit)
            # what we save needs to be list of lists
            self.crit = [xcrit]
            self.caus = [ucaus]

        elif mode=='tile2':

            if self.griddone==False:
                print("Error: plotcrit with mode=='tile2' requires tiling to be complete")
                return
            # get endpoints of all segments
            tmp = self.imgpts[self.edges]
            xA = tmp[:,0]
            xB = tmp[:,1]
            tmp = self.minv[self.edges]
            mA = tmp[:,0]
            mB = tmp[:,1]
            # find segments where magnification changes sign
            indx = np.where(mA*mB<0)[0]
            # use root finding on each relevant segment
            xcrit = []
            for i in indx:
                wtmp = fsolve(self.tile2_func,0.5,args=(xA[i],xB[i]))[0]
                xtmp = (1.0-wtmp)*xA[i] + wtmp*xB[i]
                xcrit.append(xtmp)
            xcrit = np.array(xcrit)
            # find corresponding caustic points
            ucaus,A = self.lenseqn(xcrit)
            # what we save needs to be list of lists
            self.crit = [xcrit]
            self.caus = [ucaus]

        # now make the figure
        f,ax = plt.subplots(1,2,figsize=(10,5))
        if pointtype=='line':
            for x in self.crit: ax[0].plot(x[:,0],x[:,1])
            for u in self.caus: ax[1].plot(u[:,0],u[:,1])
        else:
            for x in self.crit: ax[0].plot(x[:,0],x[:,1],pointtype)
            for u in self.caus: ax[1].plot(u[:,0],u[:,1],pointtype)
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        if len(title)>0:
            ax[0].set_title('image plane - '+title)
        else:
            ax[0].set_title('image plane')
        ax[1].set_title('source plane')
        f.tight_layout()
        if show==False:
            plt.close()
        elif len(file)==0:
            f.show()
        else:
            f.savefig(file,bbox_inches='tight')

        # update flag indicating that crit/caus have been computed
        self.critdone = True

    def tile2_func(self,w,xA,xB):
        x = (1.0-w)*xA + w*xB
        u,A = self.lenseqn(x)
        return np.linalg.det(A)

    ##################################################################
    # general purpose plotter, with options to plot the grid,
    # critical curve(s) and caustic(s), and source/image combinations
    ##################################################################

    def plot(self,imgrange=[],srcrange=[],plotgrid=False,plotcrit='black',
             src=[],title='',file=''):

        f,ax = plt.subplots(1,2,figsize=(12,6))

        # plot the grid lines
        if plotgrid:
            if self.griddone==False: self.tile()
            ax[0].add_collection(copy.copy(self.plotimg))
            ax[1].add_collection(copy.copy(self.plotsrc))

        # plot the critical curve(s) and caustic(s)
        if len(plotcrit)>0:
            # compute crit/caus if needed
            if self.critdone==False: self.plotcrit(show=False)
            for x in self.crit:
                ax[0].plot(x[:,0],x[:,1],color=plotcrit)
            for u in self.caus:
                ax[1].plot(u[:,0],u[:,1],color=plotcrit)

        # if any sources are specified, solve the lens equation
        # and plot the source(s) and images
        if len(src)>=2:
            src = np.array(src)
            if src.ndim==1: src = np.array([src])
            color_list = iter(cm.hsv(np.linspace(0,1,len(src)+1)))
            for u in src:
                imgarr,muarr = self.findimg(u)
                color = next(color_list)
                ax[0].plot(imgarr[:,0],imgarr[:,1],'.',color=color)
                ax[1].plot(u[0],u[1],'.',color=color)

        # adjust and annotate
        if len(imgrange)>=4:
            ax[0].set_xlim([imgrange[0],imgrange[1]])
            ax[0].set_ylim([imgrange[2],imgrange[3]])
        if len(srcrange)>=4:
            ax[1].set_xlim([srcrange[0],srcrange[1]])
            ax[1].set_ylim([srcrange[2],srcrange[3]])
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        if len(title)==0:
            ax[0].set_title('image plane')
        else:
            ax[0].set_title(r'image plane - '+title)
        ax[1].set_title('source plane')
        if len(file)==0:
            f.show()
        else:
            f.savefig(file,bbox_inches='tight')

