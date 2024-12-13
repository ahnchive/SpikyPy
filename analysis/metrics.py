import numpy as np

def FSI(avg_face_response, avg_obj_response):
    return (avg_face_response-avg_obj_response)/(avg_face_response+avg_obj_response)