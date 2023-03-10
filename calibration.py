import os
import pickle

import cv2 as cv
import numpy as np


CellWidth = 8
CellHeight = 6
tileSize = 115

fourCornerCoordinates = []
manual_points_entered = False

# Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPoints = []  # 2d points in image plane.
objP = np.zeros((CellWidth * CellHeight, 3), np.float32)
objP[:, :2] = np.mgrid[0:CellWidth, 0:CellHeight].T.reshape(-1, 2) * tileSize


def interpolate_grid(coordinates, image):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows, cols, ch = image.shape
    pts1 = np.float32(coordinates)
    pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    p2 = pts2[1]
    p3 = pts2[2]
    p4 = pts2[3]

    horizontalVector = np.subtract(p4, p2)
    verticalVector = np.subtract(p4, p3)

    grid = []
    for x in range(CellHeight):
        alpha = x / (CellHeight - 1)
        x_offset = (horizontalVector * alpha)[1:]
        for y in range(CellWidth):
            beta = y / (CellWidth - 1)
            y_offset = (verticalVector * beta)[:1]
            grid.append(tuple((y_offset, x_offset)))

    _, IM = cv.invert(M)
    reprojectedPoints = []
    for point in grid:
        x1, y1 = point
        coord = [x1, y1] + [1]
        P = np.array(coord, dtype=object)
        x, y, z = np.dot(IM, P)
        # Divide x and y by z to get 2D values
        reproj_x = x / z
        reproj_y = y / z
        reprojectedPoints.append((reproj_x, reproj_y))

    global corners2
    corners2 = np.reshape(reprojectedPoints, (48, 1, 2))

    objPoints.append(objP)
    imgPoints.append(corners2)
    cv.drawChessboardCorners(image, (CellWidth, CellHeight), corners2, True)
    print("Draw")
    global manual_points_entered
    manual_points_entered = True

def click_event(event, x, y, flag, params):
    global fourCornerCoordinates

    if event == cv.EVENT_LBUTTONDOWN:

        if len(fourCornerCoordinates) < 4:
            fourCornerCoordinates.append([x, y])

        if len(fourCornerCoordinates) == 4:
            cv.destroyAllWindows()
            interpolate_grid(fourCornerCoordinates, params)
            fourCornerCoordinates = []

        img = cv.circle(params, (int(x), int(y)), 5, (255, 0, 0), 2)
        cv.imshow('img', img)

def findCorners(sampleImage):
    global manual_points_entered
    global corners2

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    manual_points_entered = False
    gray = cv.cvtColor(sampleImage, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (CellWidth, CellHeight), None)
    if ret:
        cv.imshow('img', sampleImage)
        cv.setMouseCallback('img', click_event, param=sampleImage)
        while not manual_points_entered:
            cv.imshow('img', sampleImage)
            cv.waitKey(500)
        # objPoints.append(objP)
        # corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # imgPoints.append(corners2)
        # cv.drawChessboardCorners(sampleImage, (CellWidth, CellHeight), corners2, ret)
        # cv.imshow('img', sampleImage)
        # cv.waitKey(50)
    # If fail to find the corners automatically wait for user manual input
    if not ret:
        cv.imshow('img', sampleImage)
        cv.setMouseCallback('img', click_event, param=sampleImage)
        while not manual_points_entered:
            cv.imshow('img', sampleImage)
            cv.waitKey(500)

def loadIntrinsics():

    #path = os.path.join("4persons", "extrinsics")
    fileName = "camera_matrix.npz"
    with np.load(fileName) as file:
        mtx, d = [file[j] for j in ['mtx', 'dist']]

    return mtx, d


def draw(image, corners, imgPts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(image, corner, tuple(imgPts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv.line(image, corner, tuple(imgPts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv.line(image, corner, tuple(imgPts[2].ravel().astype(int)), (0, 0, 255), 5)
    cv.imshow("img", img)
    cv.waitKey(0)


def calibrateExtrinsic():
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * tileSize
    mtx, dist = loadIntrinsics()
    path = os.path.join("4persons", "extrinsics")

    for i in range(4):
        videoName = "video" + str(i + 1) + ".avi"
        videoPath = os.path.join(path, videoName)
        video = cv.VideoCapture(videoPath)
        success, frame = video.read()
        findCorners(frame)
        ret, rotation, translation = cv.solvePnP(objP, corners2, mtx, dist)

        imgPts, jac = cv.projectPoints(axis, rotation, translation, mtx, dist)
        draw(frame, corners2, imgPts)

        output = "camera_extrinsics" + str(i+1)
        #np.savez(output, rvec=rotation, tvec=translation)

    cv.destroyAllWindows()


def saveFrame():
    path = os.path.join("4persons", "video")

    for i in range(4):
        videoName = "video" + str(i + 1) + ".avi"
        videoPath = os.path.join(path, videoName)
        video = cv.VideoCapture(videoPath)
        video.set(cv.CAP_PROP_POS_FRAMES, 510)
        success,  frame = video.read()
        if success and i == 3:
            #frameName =
            #cv.imwrite(("frame" + str(i+1) + ".png"), frame)
            print("Done")

def createLookupTable():
    cameraLookupTable = {}
    intrinsicMatrix, dist = loadIntrinsics()

    # Define the range of the cube
    Xl = -10
    Xh = 23
    Yl = -19
    Yh = 14
    Zl = 2
    Zh = -16

    #voxelCoordinates = []

    # #frameName = "frame" + str(i + 1) + ".png"
    # frame = cv.imread("frame1.png")
    # #output = "camera_extrinsics" + str(i + 1) + ".npz"
    # with np.load("camera_extrinsics1.npz") as file:
    #     rotation, translation = [file[i] for i in ['rvec', 'tvec']]
    #
    # for x in np.arange(Xl, Xh, 0.5):
    #     for y in np.arange(Yl, Yh, 0.5):
    #         for z in np.arange(Zh, Zl, 0.5):
    #             output = []
    #             # Get the projected point of the voxel position.
    #             voxelPoint = np.float32((x, y, z)) * tileSize
    #             voxelCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
    #             voxelCoordinates.append(voxelCoordinate)
    #
    #             fx = int(voxelCoordinate[0][0][0])
    #             fy = int(voxelCoordinate[0][0][1])
    #
    #             Xc = voxelPoint[0]
    #             Yc = voxelPoint[1]
    #             Zc = voxelPoint[2]
    #
    #             output.append((fy, fx))
    #
    #             # Store 2d points as key and array of voxels as value
    #             if (Xc, Yc, Zc) in cameraLookupTable:
    #                 cameraLookupTable[(Xc, Yc, Zc)].append((fy, fx))
    #             else:
    #                 cameraLookupTable[(Xc, Yc, Zc)] = output

    # Draw the voxels for confirmation.
    # for voxel in voxelCoordinates:
    #     x = int(voxel[0][0][0])
    #     y = int(voxel[0][0][1])
    #     # b, g, r = color[i][(x, y)]
    #
    #     img = cv.circle(frame, (int(voxel[0][0][0]), int(voxel[0][0][1])), 1, (255, 0, 0), 2)
    # cv.imshow('img', img)
    # cv.waitKey(500)

    # for i in range(4):
    #     voxelCoordinates = []
    #
    #     frameName = "frame" + str(i+1) + ".png"
    #     frame = cv.imread(frameName)
    #     output = "camera_extrinsics" + str(i+1) + ".npz"
    #     with np.load(output) as file:
    #         rotation, translation = [file[i] for i in ['rvec', 'tvec']]
    #
    #     for x in np.arange(Xl, Xh, 0.5):
    #         for y in np.arange(Yl, Yh, 0.5):
    #             for z in np.arange(Zh, Zl, 0.5):
    #                 # Get the projected point of the voxel position.
    #                 voxelPoint = np.float32((x, y, z)) * tileSize
    #                 voxelCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
    #                 voxelCoordinates.append(voxelCoordinate)
    #
    #                 fx = int(voxelCoordinate[0][0][0])
    #                 fy = int(voxelCoordinate[0][0][1])
    #
    #                 Xc = voxelPoint[0]
    #                 Yc = voxelPoint[1]
    #                 Zc = voxelPoint[2]
    #                 # Store 2d points as key and array of voxels as value
    #                 if (fy, fx) in cameraLookupTable:
    #                     cameraLookupTable[(fy, fx)].append((Xc, Yc, Zc, i))
    #                 else:
    #                     cameraLookupTable[(fy, fx)] = [(Xc, Yc, Zc, i)]
    #
    #     #Draw the voxels for confirmation.
    #     for voxel in voxelCoordinates:
    #         x = int(voxel[0][0][0])
    #         y = int(voxel[0][0][1])
    #         #b, g, r = color[i][(x, y)]
    #
    #         img = cv.circle(frame, (int(voxel[0][0][0]), int(voxel[0][0][1])), 1, (255, 0, 0), 2)
    #     cv.imshow('img', img)
    #     cv.waitKey(500)
    # #
    with open('lookupTable.pickle', 'wb') as handle:
        pickle.dump(cameraLookupTable, handle, protocol=pickle.HIGHEST_PROTOCOL)

