# pygravlens

class lensplane:

    ##################################################################
    # initialization
    # - parr = [[x0,y0,...], [x1,y1,...], ...]
    # - kappa,gamma_c,gamma_s = convergence and shear
    # - Dl_Ds is the ratio (lens distance)/(source distance)
    ##################################################################

    def __init__(self,ID,parr=[],kappa=0,gammac=0,gammas=0,Dl_Ds=0.5):
