import numpy as np
EPSILON=1e-15

def FSI(avg_face_response, avg_obj_response):
    return (avg_face_response-avg_obj_response)/(avg_face_response+avg_obj_response+EPSILON)