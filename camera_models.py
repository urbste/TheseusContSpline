
import torch
import theseus as th

class CameraIntrinsics:
    def __init__(self, fx, fy, cx, cy):

        self.K = th.Vector(tensor=torch.eye(1,3,3).float())
        self.K[:,0,0] = fx
        self.K[:,1,1] = fy
        self.K[:,0,1] = cx
        self.K[:,1,2] = cy

    def reproject(self, X_in_cam):
        
        X_image_plane = self.K @ X_in_cam
        return X_image_plane / X_image_plane[...,-1]


