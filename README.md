# pygravlens

New python version of Keeton's gravlens software.

Dependences:
- numpy
- scipy
- matplotlib
- shapely (https://pypi.org/project/Shapely/)

Current mass models:
- point mass ('ptmass')
- Singular Isothermal Sphere ('SIS')

A fundamental unit is a lens plane:

    class lensplane(ID,parr=[],kappa=0,gammac=0,gammas=0,Dl_Ds=0.5):
    # - parr = [[x0,y0,...], [x1,y1,...], ...]
    # - kappa,gamma_c,gamma_s = convergence and shear
    # - Dl_Ds is the ratio (lens distance)/(source distance);
    #   used only in multiplane lensing

One or more lens planes can be combined into a lens model:

    class lensmodel(plane_list,xtol=1.0e-5,position_mode='obs'):
    # - plane_list is a list of lensplane structures
    # - xtol is the tolerance used for finding images
    # - position_mode is important for multiplane lensing;
    #   + 'obs' indicates that the specified positions are observed,
    #     so the intrinsic positions must account for foreground bending
    #   + 'fix' indicates that the specified positions are fixed in space

Here are some key routines:

    ##################################################################
    # report some key information about the model
    ##################################################################
    lensmodel.info()

    ##################################################################
    # commands to specify the grid:
    # - maingrid is Cartesian
    # - galgrid is polar grid(s) centered on the mass component(s)
    ##################################################################
    lensmodel.maingrid(xlo,xhi,nx,ylo,yhi,ny)
    lensmodel.galgrid(rlo,rhi,nr,ntheta)

    ##################################################################
    # compute the tiling
    ##################################################################
    lensmodel.tile()
    
    ##################################################################
    # solve the lens equation and report the image(s) for a given
    # source position or set of source positions
    ##################################################################f
    lensmodel.findimg(u)

    ##################################################################
    # general purpose plotter, with options to plot the grid,
    # critical curve(s) and caustic(s), and source/image combinations
    ##################################################################
    lensmodel.plot(imgrange=[],srcrange=[],plotgrid=False,plotcrit='black',src=[],file=''):
