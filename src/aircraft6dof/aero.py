from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AeroCoefficients:
    CL0:float=.25; CL_alpha:float=5.2; CL_de:float=.45; CL_q:float=5.5
    CD0:float=.025; CD_alpha2:float=.32; CD_de2:float=.02
    CY_beta:float=-.90; CY_da:float=.08; CY_dr:float=.20
    Cl_beta:float=-.12; Cl_p:float=-.50; Cl_r:float=.20; Cl_da:float=.20; Cl_dr:float=.05
    Cm0:float=.04; Cm_alpha:float=-1.10; Cm_q:float=-12.; Cm_de:float=-1.10
    Cn_beta:float=.25; Cn_p:float=-.05; Cn_r:float=-.25; Cn_da:float=.03; Cn_dr:float=-.10

def coefficient_buildup(alpha,beta,p,q,r,V,span,chord,controls,c):
    da,de,dr=controls; V=max(float(V),1.)
    ph=p*span/(2*V); qh=q*chord/(2*V); rh=r*span/(2*V)
    return {
      "CL":c.CL0+c.CL_alpha*alpha+c.CL_de*de+c.CL_q*qh,
      "CD":c.CD0+c.CD_alpha2*alpha**2+c.CD_de2*de**2,
      "CY":c.CY_beta*beta+c.CY_da*da+c.CY_dr*dr,
      "Cl":c.Cl_beta*beta+c.Cl_p*ph+c.Cl_r*rh+c.Cl_da*da+c.Cl_dr*dr,
      "Cm":c.Cm0+c.Cm_alpha*alpha+c.Cm_q*qh+c.Cm_de*de,
      "Cn":c.Cn_beta*beta+c.Cn_p*ph+c.Cn_r*rh+c.Cn_da*da+c.Cn_dr*dr}

def aerodynamic_loads(rho,vrel_body,alpha,beta,p,q,r,geometry,controls,coeffs):
    V=np.linalg.norm(vrel_body)
    if V<=1e-8:return np.zeros(3),np.zeros(3),{"CL":0,"CD":0,"CY":0,"Cl":0,"Cm":0,"Cn":0}
    c=coefficient_buildup(alpha,beta,p,q,r,V,geometry.wing_span_m,geometry.mean_chord_m,controls,coeffs)
    qb=.5*rho*V*V; S=geometry.wing_area_m2
    fw=qb*S*np.array([-c["CD"],c["CY"],-c["CL"]])
    sa,ca=np.sin(alpha),np.cos(alpha); sb,cb=np.sin(beta),np.cos(beta)
    c_bw=np.array([[ca*cb,-ca*sb,-sa],[sb,cb,0],[sa*cb,-sa*sb,ca]])
    fb=c_bw@fw
    mb=qb*S*np.array([geometry.wing_span_m*c["Cl"],geometry.mean_chord_m*c["Cm"],geometry.wing_span_m*c["Cn"]])
    return fb,mb,c
