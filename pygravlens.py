import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from shapely.geometry import Point, Polygon
import copy


################################################################################
# UTILITIES
################################################################################

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
parameters = [x0,y0,thetaE]
"""
def calc_ptmass(parr,x):
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

        phir_r = thetaE**2/r**2
        phirr  = -phir_r
        phixx  = phir_r*sint*sint + phirr*cost*cost
        phiyy  = phir_r*cost*cost + phirr*sint*sint
        phixy  = (phirr-phir_r)*sint*cost

        alpha += np.array([ phir_r[i]*dx[i] for i in range(len(x)) ])
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
    # - kappa,gamma_c,gamma_s = convergence and shear
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
    ##################################################################

    def __init__(self,plane_list,xtol=1.0e-5,position_mode='obs'):
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

        # process distances
        self.nplane = len(plane_list)
        dtmp = []
        for plane in plane_list:
            dtmp.append(plane.Dl_Ds)
        dtmp.append(1.0)
        dtmp = np.array(dtmp)

        # make sure the planes are sorted in distance order
        indx = np.argsort(dtmp)
        darr = dtmp[indx]
        self.plane_list = []
        for i in indx[:-1]: self.plane_list.append(plane_list[i])

        # compute beta factors
        self.beta = np.zeros((self.nplane+1,self.nplane+1))
        for j in range(self.nplane+1):
            for i in range(j):
                self.beta[i,j] = ((darr[j]-darr[i])*darr[-1])/((darr[-1]-darr[i])*darr[j])

        # see if model is 3d
        self.flag3d = False
        for i in range(1,self.nplane):
            if abs(self.plane_list[i].Dl_Ds-self.plane_list[i-1].Dl_Ds)>1.0e-6: self.flag3d = True

        # process pobs and pfix
        if self.flag3d==False:
            # not 3d, so things are simple
            for plane in self.plane_list:
                plane.pobs = plane.parr[:,0:2]
                plane.pfix = plane.parr[:,0:2]
        elif self.position_mode=='obs':
            # we need to map the observed positions back to find
            # the intrinsic positions
            for i,plane in enumerate(self.plane_list):
                plane.pobs = plane.parr[:,0:2]
                plane.pfix,A = self.lenseqn(plane.pobs,stopplane=i)
                plane.parr[:,0:2] = plane.pfix
        # note: 3d with position_mode=='fix' is handled in find_centers()

    ##################################################################
    # report some key information about the model
    ##################################################################

    def info(self):
        print('number of planes:',self.nplane)
        print('maingrid:',self.maingrid_info)
        print('galgrid:',self.galgrid_info)
        if self.flag3d:
            print('model is 3d')
            print('beta:',self.beta)
            print('position mode:',self.position_mode)

    ##################################################################
    # lens equation; take an arbitrary set of image positions and return
    # the corresponding set of source positions; can handle multiplane
    # lensing; stopplane can be used to stop at some specified plane,
    # and stopplane<0 means go all the way to the source
    ##################################################################

    def lenseqn(self,xarr,stopplane=-1):
        if stopplane<0: stopplane = len(self.plane_list)
        xarr = np.array(xarr)
        # need special treatment if xarr is a single point
        if xarr.ndim==1:
            oneflag = True
            xarr = np.array([xarr])
        else:
            oneflag = False

        # structures to store everything - all positions, all planes
        xshape = list(xarr.shape[:-1])
        xall = np.zeros([self.nplane+1]+xshape+[2])
        Aall = np.zeros([self.nplane+1]+xshape+[2,2])
        alphaall = np.zeros([self.nplane+1]+xshape+[2])
        GammAall = np.zeros([self.nplane+1]+xshape+[2,2])

        # set of identity matrices for all positions
        tmp0 = np.zeros(xshape)
        tmp1 = tmp0 + 1.0
        bigI = np.moveaxis(np.array([[tmp1,tmp0],[tmp0,tmp1]]),[0,1],[-2,-1])

        # initialize first plane
        xall[0] = xarr
        Aall[0] = bigI

        # construct the z and A lists by iterating
        for j in range(stopplane):
            # compute this plane
            alpha_now,Gamma_now = self.plane_list[j].defmag(xall[j])
            # we need Gamma@A, not Gamma by itself
            Gamma_A_now = Gamma_now@Aall[j]
            # store this plane
            alphaall[j] = alpha_now
            GammAall[j] = Gamma_A_now
            # compute the lens equation
            xall[j+1] = xall[0]
            Aall[j+1] = Aall[0]
            for i in range(j+1):
                xall[j+1] = xall[j+1] - self.beta[i,j+1]*alphaall[i]
                Aall[j+1] = Aall[j+1] - self.beta[i,j+1]*GammAall[i]

        # return the desired plane
        if oneflag:
            return xall[stopplane][0],Aall[stopplane][0]
        else:
            return xall[stopplane],Aall[stopplane]

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

    def tile(self,addlevels=2,addpoints=5):
        # find the centers
        self.find_centers()
        # do the (final) tiling
        self.do_tile(addlevels=addlevels,addpoints=addpoints)
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
            for plane in self.plane_list:
                plane.pobs = plane.parr[:,0:2]
                plane.pfix = plane.parr[:,0:2]
                for p in plane.pobs: centers.append(p)
            self.centers = np.array(centers)

        else:

            # model is 3d, so we need to take care with the centers
            if self.position_mode=='obs':
                # we already processed pobs and pfix in __init__()
                for plane in self.plane_list:
                    for p in plane.pobs: centers.append(p)
                self.centers = np.array(centers)
            elif self.position_mode=='fix':
                # the specified positions are fixed, so we need
                # to solve the lens equation (for the appropriate
                # source plane) to find the corresponding observed
                # positions
                for i,plane in enumerate(self.plane_list):
                    plane.pfix = plane.parr[:,0:2]
                    if i==0:
                        # for first plane, just use pfix
                        for p in plane.pfix: centers.append(p)
                    else:
                        # we need to tile with what we have so far
                        self.do_tile(stopplane=i)
                        # solve the (intermediate) lens equation to find
                        # the observed position(s) of the center(s)
                        for pfix in plane.pfix:
                            pobs,mu = self.findimg(pfix,plane=i)
                            for p in pobs: centers.append(p)
                    self.centers = np.array(centers)
            else:
                print('Error: unknown position_mode')
                return

    ##################################################################
    # internal: this is the workhorse that does the tiling
    ##################################################################

    def do_tile(self,stopplane=-1,addlevels=2,addpoints=5):

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

        # positions in image plane, from maingrid and galgrid depending on what is available
        if len(self.maingrid_pts)>0 and len(self.galgrid_pts)>0:
            self.imgpts = np.concatenate((self.maingrid_pts,self.galgrid_pts),axis=0)
        elif len(self.maingrid_pts)>0:
            self.imgpts = self.maingrid_pts
        else:
            self.imgpts = self.galgrid_pts
        # positions in source plane, and inverse magnifications
        u,A = self.lenseqn(self.imgpts,stopplane=stopplane)
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
                xtry = np.mean(self.imgpoly[itri,:3],axis=0)
                ans = minimize(self.findimg_func,xtry,args=(u,plane),tol=0.1*self.xtol)
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
    # plot critical curve(s) and caustic(s); the search range is taken
    # from maingrid, with the specified number of steps in each direction
    ##################################################################

    def plotcrit(self,steps=500,show=True,title='',file=''):
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
        f,ax = plt.subplots(1,2,figsize=(10,5))

        # loop over all "segments" in the contour plot
        x0 = np.array([xlo,ylo])
        for v in cnt.allsegs[0]:
            # convert from pixel units to arcsec in image plane
            x = x0 + pixscale*v
            u,A = self.lenseqn(x)
            self.crit.append(x)
            self.caus.append(u)
            ax[0].plot(x[:,0],x[:,1])
            ax[1].plot(u[:,0],u[:,1])

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

    ##################################################################
    # general purpose plotter, with options to plot the grid,
    # critical curve(s) and caustic(s), and source/image combinations
    ##################################################################

    def plot(self,imgrange=[],srcrange=[],plotgrid=False,plotcrit='black',
             src=[],file=''):

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
        ax[0].set_title('image plane')
        ax[1].set_title('source plane')
        if len(file)==0:
            f.show()
        else:
            f.savefig(file,bbox_inches='tight')


