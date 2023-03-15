import calibration as calibrate
import background_subtraction as bs
import os
import pickle
import cv2 as cv
import numpy as np

tileSize = 115


# Generate lookup table for XOR method
def xorLookupTable():
    cameraLookupTable = {}

    # Define the range of the cube
    Xl = -10
    Xh = 23
    Yl = -19
    Yh = 14
    Zl = 2
    Zh = -16

    for i in range(4):

        voxelCoordinates = []

        # Read in the foreground image.
        camFolder = os.path.join("4persons", "video")
        videoName = "video" + str(i + 1) + ".avi"
        path = os.path.join(camFolder, videoName)
        video = cv.VideoCapture(path)
        success, frame = video.read()
        cameraData = "camera_extrinsics" + str(i + 1) + ".npz"

        # read in the camera matrix
        with np.load('camera_matrix.npz') as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
        with np.load(cameraData) as file:
            rotation, translation = [file[i] for i in ['rvec', 'tvec']]

        for x in np.arange(Xl, Xh, 0.5):
            for y in np.arange(Yl, Yh, 0.5):
                for z in np.arange(Zh, Zl, 0.5):
                    # Get the projected point of the voxel position.
                    voxelPoint = np.float32((x, y, z)) * tileSize
                    voxelCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
                    fx = int(voxelCoordinate[0][0][0])
                    fy = int(voxelCoordinate[0][0][1])

                    Xc = voxelPoint[0]
                    Yc = voxelPoint[1]
                    Zc = voxelPoint[2]
                    # Store 2d points as key and array of voxels as value
                    if (fy, fx) in cameraLookupTable:
                        cameraLookupTable[(fy, fx)].append((Xc, Yc, Zc, i))
                    else:
                        cameraLookupTable[(fy, fx)] = [(Xc, Yc, Zc, i)]

    with open('xor.pickle', 'wb') as handle:
        pickle.dump(cameraLookupTable, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(cameraLookupTable)


if __name__ == '__main__':
    # calibrate.calibrateExtrinsic()
    # calibrate.saveFrame()
    # calibrate.createLookupTable()
    # bs.createBackgroundModel()
    # bs.GenerateForeground()
    # xorLookupTable()

    print("main")
