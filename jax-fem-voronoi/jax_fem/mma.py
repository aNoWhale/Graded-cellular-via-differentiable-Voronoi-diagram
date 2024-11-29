"""
Copied and modified from https://github.com/UW-ERSL/AuTO
Under GNU General Public License v3.0

Original copy from https://github.com/arjendeetman/GCMMA-MMA-Python/blob/master/Code/MMA.py

Improvement is made to avoid N^2 memory operation so that the MMA solver is more scalable.
"""
from numpy import diag as diags
from numpy.linalg import solve
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random, jacfwd, value_and_grad
from functools import partial
import time
import scipy
import sys
import os
from jax import config
from tqdm import tqdm
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import jax_fem.numpy2stl


def compute_filter_kd_tree(fe):
    """This function is created by Tianju. Not from the original code.
    We use k-d tree algorithm to compute the filter.
    revised by changkun
    """
    cell_centroids = np.mean(np.take(fe.points, fe.cells, axis=0), axis=1)
    flex_num_cells = len(fe.flex_inds)
    flex_cell_centroids = np.take(cell_centroids, fe.flex_inds, axis=0)
    V = np.sum(fe.JxW)
    avg_elem_V = V / fe.num_cells
    avg_elem_size = avg_elem_V ** (1. / fe.dim)
    rmin = 1.5 * avg_elem_size
    kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
    I = []
    J = []
    V = []
    for i in range(flex_num_cells):
        num_nbs = 20
        dd, ii = kd_tree.query(flex_cell_centroids[i], num_nbs)
        neighbors = np.take(flex_cell_centroids, ii, axis=0)
        vals = np.where(rmin - dd > 0., rmin - dd, 0.)
        I += [i] * num_nbs
        J += ii.tolist()
        V += vals.tolist()
    H_sp = scipy.sparse.csc_matrix((V, (I, J)), shape=(flex_num_cells, flex_num_cells))

    H=H_sp
    Hs = np.array(H_sp.sum(axis=1))
    # H = H_sp.todense()
    # Hs = np.sum(H, 1)
    return H, Hs


def applySensitivityFilter(ft, p, dJ, dvc):
    # changkun changed here
    # print("ft['H'].shape:", ft['H'].shape)
    # print("ft['Hs'].shape:", ft["Hs"].shape)
    # print("ft['Hs'][:,N].shape:", ft["Hs"][:,None].shape)
    # print("p.shape:", p.shape)
    # print("dvc.shape:", dvc.shape)
    # print("dJ.shape:", dJ.shape)
    # dJ = np.matmul(ft['H'], p * dJ / np.maximum(1e-3, p) / ft['Hs'][:, None])
    dJ = ft['H'].dot(p * dJ / np.maximum(1e-3, p) / ft['Hs']) # # for sparse matrix
    # dvc = np.matmul(ft['H'][None, :, :], p[None, :, :] * dvc / np.maximum(1e-3, p[None, :, :]) / ft['Hs'][None, :, None])
    temp=p * dvc / np.maximum(1e-3, p) / ft['Hs']
    dvc = ft['H'].dot(temp.squeeze(axis=0)) # for sparse matrix
    # print("..........")
    # print("dJ.shape:", dJ.shape)
    # print("dvc.shape:", dvc.shape)
    return dJ, dvc


#%% Optimizer
class MMA:
    # The code was modified from [MMA Svanberg 1987]. Please cite the paper if
    # you end up using this code.
    def __init__(self):
        self.epoch = 0;
    def resetMMACounter(self):
        self.epoch = 0;
    def registerMMAIter(self, xval, xold1, xold2):
        self.epoch += 1;
        self.xval = xval;
        self.xold1 = xold1;
        self.xold2 = xold2;
    def setNumConstraints(self, numConstraints):
        self.numConstraints = numConstraints;
    def setNumDesignVariables(self, numDesVar):
        self.numDesignVariables = numDesVar;
    def setMinandMaxBoundsForDesignVariables(self, xmin, xmax):
        self.xmin = xmin;
        self.xmax = xmax;
    def setObjectiveWithGradient(self, obj, objGrad):
        self.objective = obj;
        self.objectiveGradient = objGrad;
    def setConstraintWithGradient(self, cons, consGrad):
        self.constraint = cons;
        self.consGrad = consGrad;
    def setScalingParams(self, zconst, zscale, ylinscale, yquadscale):
        self.zconst = zconst;
        self.zscale = zscale;
        self.ylinscale = ylinscale;
        self.yquadscale = yquadscale;
    def setMoveLimit(self, movelim):
        self.moveLimit = movelim;
    def setLowerAndUpperAsymptotes(self, low, upp):
        self.lowAsymp = low;
        self.upAsymp = upp;

    def getOptimalValues(self):
        return self.xmma, self.ymma, self.zmma;
    def getLagrangeMultipliers(self):
        return self.lam, self.xsi, self.eta, self.mu, self.zet;
    def getSlackValue(self):
        return self.slack;
    def getAsymptoteValues(self):
        return self.lowAsymp, self.upAsymp;

    # Function for the MMA sub problem
    def mmasub(self, xval):
        m = self.numConstraints;
        n = self.numDesignVariables;
        iter = self.epoch;
        xmin, xmax = self.xmin, self.xmax;
        xold1, xold2 = self.xold1, self.xold2;
        f0val, df0dx = self.objective, self.objectiveGradient;
        fval, dfdx = self.constraint, self.consGrad;
        low, upp = self.lowAsymp, self.upAsymp;
        a0, a, c, d = self.zconst, self.zscale, self.ylinscale, self.yquadscale;
        move = self.moveLimit;

        epsimin = 0.0000001
        raa0 = 0.00001
        albefa = 0.1
        asyinit = 0.5
        asyincr = 1.2
        asydecr = 0.7
        eeen = np.ones((n, 1))
        eeem = np.ones((m, 1))
        zeron = np.zeros((n, 1))
        # Calculation of the asymptotes low and upp
        if iter <= 2:
            low = xval-asyinit*(xmax-xmin)
            upp = xval+asyinit*(xmax-xmin)
        else:
            zzz = (xval-xold1)*(xold1-xold2) #计算变化的方向
            factor = eeen.copy()
            factor[np.where(zzz>0)] = asyincr
            factor[np.where(zzz<0)] = asydecr
            low = xval-factor*(xold1-low)
            upp = xval+factor*(upp-xold1)
            # 渐近线范围
            lowmin = xval-10*(xmax-xmin)
            lowmax = xval-0.01*(xmax-xmin)
            uppmin = xval+0.01*(xmax-xmin)
            uppmax = xval+10*(xmax-xmin)
            low = np.maximum(low,lowmin)
            low = np.minimum(low,lowmax)
            upp = np.minimum(upp,uppmax)
            upp = np.maximum(upp,uppmin)
        # Calculation of the bounds alfa and beta
        zzz1 = low+albefa*(xval-low)
        zzz2 = xval-move*(xmax-xmin)
        zzz = np.maximum(zzz1,zzz2)
        alfa = np.maximum(zzz,xmin)
        zzz1 = upp-albefa*(upp-xval)
        zzz2 = xval+move*(xmax-xmin)
        zzz = np.minimum(zzz1,zzz2)
        beta = np.minimum(zzz,xmax)
        # Calculations of p0, q0, P, Q and b
        xmami = xmax-xmin
        xmamieps = 0.00001*eeen
        xmami = np.maximum(xmami,xmamieps)
        xmamiinv = eeen/xmami
        #xval距离渐近线的距离
        ux1 = upp-xval
        ux2 = ux1*ux1
        xl1 = xval-low
        xl2 = xl1*xl1
        uxinv = eeen/ux1
        xlinv = eeen/xl1

        p0 = np.maximum(df0dx,0) #目标梯度的正部分
        q0 = np.maximum(-df0dx,0) #目标梯度的负部分
        # 引入微小扰动pq0防止梯度过小
        pq0 = 0.001*(p0+q0)+raa0*xmamiinv
        p0 = p0+pq0
        q0 = q0+pq0
        #权重更新，使得p,q与距离有关
        p0 = p0*ux2
        q0 = q0*xl2
        P = np.zeros((m,n)) ## @@ make sparse with scipy?
        Q = np.zeros((m,n)) ## @@ make sparse with scipy?
        #约束函数的正负梯度
        P = np.maximum(dfdx,0)
        Q = np.maximum(-dfdx,0)
        PQ = 0.001*(P+Q)+raa0*np.dot(eeem,xmamiinv.T)
        P = P+PQ
        Q = Q+PQ

        # P = (diags(ux2.flatten(),0).dot(P.T)).T
        # Q = (diags(xl2.flatten(),0).dot(Q.T)).T
        #最终形状应该是 m,n 表示每个约束对每个变量的正负梯度
        P = ux2.T*P
        Q = xl2.T*Q

        b = (np.dot(P,uxinv)+np.dot(Q,xlinv)-fval) # m,1
        # Solving the subproblem by a primal-dual Newton method
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,\
                                                      beta,p0,q0,P,Q,a0,a,b,c,d)
        # Return values
        self.xmma, self.ymma, self.zmma = xmma, ymma, zmma;
        self.lam, self.xsi, self.eta, self.mu, self.zet = lam,xsi,eta,mu,zet;
        self.slack = s;
        self.lowAsymp, self.upAsymp = low, upp;


def subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d):
    een = np.ones((n,1))
    eem = np.ones((m,1))
    epsi = 1
    epsvecn = epsi*een
    epsvecm = epsi*eem
    # 拉格朗日乘子初始化
    x = 0.5*(alfa+beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een/(x-alfa)
    xsi = np.maximum(xsi,een)
    eta = een/(beta-x)
    eta = np.maximum(eta,een)
    mu = np.maximum(eem,0.5*c)

    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    # Start while epsi>epsimin
    while epsi > epsimin: #控制误差参数 epsi 的减小，用于逐步提高计算精度
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        #计算约束残差
        uxinv1 = een/ux1
        xlinv1 = een/xl1
        plam = p0+np.dot(P.T,lam)
        qlam = q0+np.dot(Q.T,lam)
        gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1) ## 梯度向量

        dpsidx = plam/ux2-qlam/xl2
        rex = dpsidx-xsi+eta
        rey = c+d*y-mu-lam
        rez = a0-zet-np.dot(a.T,lam)
        relam = gvec-a*z-y+s-b
        rexsi = xsi*(x-alfa)-epsvecn
        reeta = eta*(beta-x)-epsvecn
        remu = mu*y-epsvecm
        rezet = zet*z-epsi
        res = lam*s-epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis = 0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis = 0)
        residu = np.concatenate((residu1, residu2), axis = 0)
        residunorm = np.sqrt((np.dot(residu.T,residu)).item())
        residumax = np.max(np.abs(residu))
        ittt = 0
        # Start while (residumax>0.9*epsi) and (ittt<200)
        while (residumax > 0.9*epsi) and (ittt < 200): #使得残差 residumax 满足精度要求
            ittt = ittt+1
            itera = itera+1
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            plam = p0+np.dot(P.T,lam)
            qlam = q0+np.dot(Q.T,lam)
            gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)

            # GG = (diags(uxinv2.flatten(),0).dot(P.T)).T-(diags\
            #                          (xlinv2.flatten(),0).dot(Q.T)).T
            GG = uxinv2.T*P - xlinv2.T*Q

            dpsidx = plam/ux2-qlam/xl2
            #计算变量步长
            delx = dpsidx-epsvecn/(x-alfa)+epsvecn/(beta-x)
            dely = c+d*y-lam-epsvecm/y
            delz = a0-np.dot(a.T,lam)-epsi/z
            dellam = gvec-a*z-y-b+epsvecm/lam
            diagx = plam/ux3+qlam/xl3
            diagx = 2*diagx+xsi/(x-alfa)+eta/(beta-x)
            diagxinv = een/diagx
            diagy = d+mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam+diagyinv
            # Start if m<n
            if m < n:
                blam = dellam+dely/diagy-np.dot(GG,(delx/diagx))
                bb = np.concatenate((blam,delz),axis = 0)
                
                # Alam = np.asarray(diags(diaglamyi.flatten(),0) \
                #     +(diags(diagxinv.flatten(),0).dot(GG.T).T).dot(GG.T))
                Alam = diags(diaglamyi.flatten(),0) + (diagxinv.T*GG).dot(GG.T)

                AAr1 = np.concatenate((Alam,a),axis = 1)
                AAr2 = np.concatenate((a,-zet/z),axis = 0).T
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                solut = solve(AA,bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]
                dx = -delx/diagx-np.dot(GG.T,dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam+dely/diagy
                Axx = np.asarray(diags(diagx.flatten(),0) \
                    +(diags(diaglamyiinv.flatten(),0).dot(GG).T).dot(GG))
                azz = zet/z+np.dot(a.T,(a/diaglamyi))
                axz = np.dot(-GG.T,(a/diaglamyi))
                bx = delx+np.dot(GG.T,(dellamyi/diaglamyi))
                bz = delz-np.dot(a.T,(dellamyi/diaglamyi))
                AAr1 = np.concatenate((Axx,axz),axis = 1)
                AAr2 = np.concatenate((axz.T,azz),axis = 1)
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                bb = np.concatenate((-bx,-bz),axis = 0)
                solut = solve(AA,bb)
                dx = solut[0:n]
                dz = solut[n:n+1]
                dlam = np.dot(GG,dx)/diaglamyi-dz*(a/diaglamyi)\
                    +dellamyi/diaglamyi
                # End if m<n
            dy = -dely/diagy+dlam/diagy
            dxsi = -xsi+epsvecn/(x-alfa)-(xsi*dx)/(x-alfa)
            deta = -eta+epsvecn/(beta-x)+(eta*dx)/(beta-x)
            dmu = -mu+epsvecm/y-(mu*dy)/y
            dzet = -zet+epsi/z-zet*dz/z
            ds = -s+epsvecm/lam-(s*dlam)/lam
            xx = np.concatenate((y,z,lam,xsi,eta,mu,zet,s),axis = 0)
            dxx = np.concatenate((dy,dz,dlam,dxsi,deta,dmu,dzet,ds),axis = 0)
            """在这里加了个小常数"""
            # print(f"dxx:{dxx}")
            # print(f"xx:{xx}")
            stepxx = -1.01*dxx/(xx+ 1e-10)
            stmxx = np.max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa,stmbeta)
            stmalbexx = max(stmalbe,stmxx)
            stminv = max(stmalbexx,1.0)
            steg = 1.0/stminv
            #
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            #
            itto = 0
            resinew = 2*residunorm

            # Start: while (resinew>residunorm) and (itto<50)
            while (resinew > residunorm) and (itto < 50):
                itto = itto+1
                x = xold+steg*dx
                y = yold+steg*dy
                z = zold+steg*dz
                lam = lamold+steg*dlam
                xsi = xsiold+steg*dxsi
                eta = etaold+steg*deta
                mu = muold+steg*dmu
                zet = zetold+steg*dzet
                s = sold+steg*ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0+np.dot(P.T,lam)
                qlam = q0+np.dot(Q.T,lam)
                gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
                dpsidx = plam/ux2-qlam/xl2
                rex = dpsidx-xsi+eta
                rey = c+d*y-mu-lam
                rez = a0-zet-np.dot(a.T,lam)
                relam = gvec-np.dot(a,z)-y+s-b
                rexsi = xsi*(x-alfa)-epsvecn
                reeta = eta*(beta-x)-epsvecn
                remu = mu*y-epsvecm
                rezet = np.dot(zet,z)-epsi
                res = lam*s-epsvecm
                residu1 = np.concatenate((rex,rey,rez),axis = 0)
                residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res), \
                                         axis = 0)
                residu = np.concatenate((residu1,residu2),axis = 0)
                resinew = np.sqrt(np.dot(residu.T,residu))
                steg = steg/2
                # End: while (resinew>residunorm) and (itto<50)

            residunorm = resinew.copy()
            residumax = max(abs(residu))
            steg = 2*steg
            # End: while (residumax>0.9*epsi) and (ittt<200)
        epsi = 0.1*epsi
        # End: while epsi>epsimin

    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma


###changkun sun rewritten here
def optimize(fe, p_ini, optiPara, objectiveHandle, consHandle, numConstraints, generate_rho):
    H, Hs = compute_filter_kd_tree(fe) # related with rho
    ft = {'H': H, 'Hs': Hs}
    p=p_ini
    m = numConstraints
    n = len(p.reshape(-1))

    mma = MMA()
    mma.setNumConstraints(numConstraints)
    mma.setNumDesignVariables(n)

    bound_low= optiPara["bound_low"][optiPara["paras_at"][0]:optiPara["paras_at"][1]] if "paras_at" in optiPara else optiPara["bound_low"]
    bound_up= optiPara["bound_up"][optiPara["paras_at"][0]:optiPara["paras_at"][1]] if "paras_at" in optiPara else optiPara["bound_up"]
    mma.setMinandMaxBoundsForDesignVariables(bound_low,bound_up)

    xval = p.reshape(-1)[:, None]
    xold1, xold2 = xval.copy(), xval.copy()
    mma.registerMMAIter(xval, xold1, xold2)
    mma.setLowerAndUpperAsymptotes(np.ones((n, 1)), np.ones((n, 1)))
    mma.setScalingParams(1.0, np.zeros((m, 1)), 10000 * np.ones((m, 1)), np.zeros((m, 1)))
    mma.setMoveLimit(optiPara['movelimit'])
    with tqdm(total=optiPara['maxIters']) as pbar:
        loop = 0
        while loop < optiPara['maxIters']:
            loop += 1
            pbar.update(1)
            print(f"MMA solver...")
            rho = generate_rho(optiPara, p, epoch=loop)
            jax_fem.numpy2stl.generate_stl_from_matrix(rho,threshold=0.5,cube_size=1,filename=f'{loop + optiPara["lastIters"]}')
            ####render windows and save fig
            sites=p[0:optiPara["sites_num"]*optiPara["dim"]].reshape(optiPara["sites_num"],optiPara["dim"])
            plt.clf()
            plt.imshow(rho,cmap='viridis')
            if "sites_boundary" in optiPara:
                osites = np.array(optiPara["sites_boundary"])
                plt.scatter(osites[:, 1]/optiPara["resolution"], osites[:, 0]/optiPara["resolution"], c="w", marker="+")
            plt.title(f"loop:{loop + optiPara['lastIters']}/{optiPara['maxIters'] + optiPara['lastIters']}")
            plt.scatter(sites[:,1]/optiPara["resolution"],sites[:,0]/optiPara["resolution"],color='r',marker='+')
            plt.colorbar()
            plt.draw()
            plt.savefig(f'data/vtk/{loop + optiPara["lastIters"]}.png', dpi=600, bbox_inches='tight')
            plt.pause(0.01)

            rho=rho.flatten()[:, None]
            assert rho.shape[1]==1
            J, dJ = objectiveHandle(rho) # get from rho = fun(p)
            vc, dvc = consHandle(rho) # get from rho

            dJ_drho, dvc_drho = applySensitivityFilter(ft, rho, dJ, dvc)
            def rho_faltten(op, p,epoch):
                fl=generate_rho(op,p,epoch=epoch)
                return fl.flatten()
            """这里控制着对哪个参数求导"""
            drho_dp = jax.jacfwd(rho_faltten, argnums=1)(optiPara, p, epoch=loop)
            dJ= np.dot(dJ_drho.T, drho_dp)
            dvc = np.dot(dvc_drho.squeeze().T, drho_dp)

            # J, dJ = J, dJ.reshape(-1)[:, None]
            # vc, dvc = vc[:, None], dvc.reshape(dvc.shape[0], -1)
            J, dJ = J, dJ.reshape(-1)[:, None]
            vc, dvc = vc[:, None], dvc.squeeze()[None,:]
            print(f"J.shape = {J.shape}")
            print(f"dJ.shape = {dJ.shape}")
            print(f"vc.shape = {vc.shape}")
            print(f"dvc.shape = {dvc.shape}")

            J, dJ, vc, dvc = np.array(J), np.array(dJ), np.array(vc), np.array(dvc)

            start = time.time()

            mma.setObjectiveWithGradient(J, dJ)
            mma.setConstraintWithGradient(vc, dvc)
            mma.mmasub(xval)
            xmma, _, _ = mma.getOptimalValues()

            xold2 = xold1.copy()
            xold1 = xval.copy()
            xval = xmma.copy()

            mma.registerMMAIter(xval, xold1, xold2)
            p = xval.reshape(p.shape)  # 更新控制参数

            end = time.time()

            time_elapsed = end - start

            print(f"MMA took {time_elapsed} [s]")
            print(f'Iter {loop:d}; J {J:.5f}; constraint {vc}\n\n\n')

    return p,J,rho
