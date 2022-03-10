# pygravlens

New python version of Keeton's gravlens software.

Dependences:
- numpy
- scipy
- matplotlib
- shapely (https://pypi.org/project/Shapely/)

A fundamental unit is a lens plane:

    class lensplane(ID,parr=[],kappa=0,gammac=0,gammas=0,Dl_Ds=0.5)
    # initialization
    # - parr = [[x0,y0,...], [x1,y1,...], ...]
    # - kappa,gamma_c,gamma_s = convergence and shear
    # - Dl_Ds is the ratio (lens distance)/(source distance);
    #   used only in multiplane lensing
