import numpy as np
from .mathutils import dcm_body_to_ned_from_quat

def body_to_ned(v_body,q_bn): return dcm_body_to_ned_from_quat(q_bn)@np.asarray(v_body,dtype=float)
def ned_to_body(v_ned,q_bn): return dcm_body_to_ned_from_quat(q_bn).T@np.asarray(v_ned,dtype=float)
