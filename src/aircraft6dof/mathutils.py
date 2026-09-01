from __future__ import annotations
import numpy as np

def normalize_quaternion(q):
    q=np.asarray(q,dtype=float); n=np.linalg.norm(q)
    if n<=1e-15 or not np.isfinite(n): raise ValueError("invalid quaternion")
    q=q/n
    return -q if q[0]<0 else q

def quat_multiply(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

def quat_from_euler321(phi,theta,psi):
    c1,s1=np.cos(phi/2),np.sin(phi/2); c2,s2=np.cos(theta/2),np.sin(theta/2); c3,s3=np.cos(psi/2),np.sin(psi/2)
    return normalize_quaternion(np.array([c1*c2*c3+s1*s2*s3,s1*c2*c3-c1*s2*s3,c1*s2*c3+s1*c2*s3,c1*c2*s3-s1*s2*c3]))

def dcm_body_to_ned_from_quat(q):
    w,x,y,z=normalize_quaternion(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def euler321_from_quat(q):
    w,x,y,z=normalize_quaternion(q)
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])
