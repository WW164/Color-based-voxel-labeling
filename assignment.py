import glm
import numpy as np
import cv2 as cv
import os
from numpy import load
import pickle
import background_subtraction as bs
import time
import calibration as calibrate
import matplotlib.pyplot as plt

block_size = 1
frameCellWidth = 1000
frameCellHeight = 1000
tileSize = 115
frameIndex = 1
pixels = []
voxels = []
previousForegroundImages = []
voxelsOnCam = {0: [], 1: [], 2: [], 3: []}


def loadPickle(type):
    if type == 'voxels':
        with open('lookupTable.pickle', 'rb') as handle:
            lookupTable = pickle.load(handle)
    else:
        with open('xor.pickle', 'rb') as handle:
            lookupTable = pickle.load(handle)

    return lookupTable


def getData():
    rvecs = []
    tvecs = []

    for i in range(4):
        fileName = "camera_extrinsics" + str(i+1) + ".npz"
        data = load(fileName)
        lst = data.files
        for item in lst:
            if item == 'rvec':
                rvecs.append(data[item])
            if item == 'tvec':
                tvecs.append(np.divide(data[item], tileSize))

    return rvecs, tvecs


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def GetForegroundValue(foregroundImages, index, coords):

    x, y = coords

    img = foregroundImages[index]
    maxX, maxY = img.shape
    if x < maxX and y < maxY:
        return np.linalg.norm(img[int(x), int(y)]) > 1


def GenerateForeground():
    global frameIndex
    foregroundImages = []

    for i in range(4):
        filepath = os.path.join("4persons", "video")
        videoName = "video" + str(i + 1) + ".avi"
        videoPath = os.path.join(filepath, videoName)
        video = cv.VideoCapture(videoPath)
        totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)
        #check for valid frame number
        if frameIndex >= 0 & frameIndex <= totalFrames:
            video.set(cv.CAP_PROP_POS_FRAMES, frameIndex)
            ret, frame = video.read()
            result = bs.backgroundSubtraction(frame, i)
            foregroundImages.append(result)
        else:
            print("ERROR: Invalid Frame")
    return foregroundImages


def finaliseVoxels(width, height, depth):
    data, colors = [], []
    onVoxels = set.intersection(set(voxelsOnCam[0]), set(voxelsOnCam[1]), set(voxelsOnCam[2]), set(voxelsOnCam[3]))

    for vox in onVoxels:

        Xc = vox[0]
        Yc = vox[1]
        Zc = vox[2]

        scalar = 0.01
        fixedPoint = (Xc * scalar, -Zc * scalar, Yc * scalar)
        data.append((fixedPoint[0],
                     fixedPoint[1],
                     fixedPoint[2]))
        colors.append([fixedPoint[0] / width, fixedPoint[1] / depth, fixedPoint[2] / height])

    return data, colors


def FirstFrameVoxelPositions(foregroundImages, width, height, depth):
    global voxelsOnCam
    voxelsOnCam = {0: [], 1: [], 2: [], 3: []}
    for pixel in pixels:
        x, y = pixel
        for j in range(len(foregroundImages)):
            maxX, maxY = foregroundImages[j].shape
            if abs(x) < maxX and abs(y) < maxY:
                if np.linalg.norm(foregroundImages[j][pixel]) > 1:
                    for voxel in pixels[pixel]:
                        if voxel[3] == j:
                            vCoord = (voxel[0], voxel[1], voxel[2])
                            voxelsOnCam[j].append(vCoord)
    data, colors = finaliseVoxels(width, height, depth)

    return data, colors


def XORFrameVoxelPositions(currImgs, prevImgs, width, height, depth):
    global voxelsOnCam, pixels
    start_time = time.time()
    for i in range(len(currImgs)):
        mask_xor = cv.bitwise_xor(currImgs[i], prevImgs[i])
        nowOnPixels = cv.bitwise_and(currImgs[i], mask_xor)
        nowOnPixels = np.argwhere(nowOnPixels == 255)
        for pixel in nowOnPixels:
            x, y = pixel
            if (x, y) in pixels:
                for values in pixels[(x, y)]:
                    if values[3] == i:
                        vCoord = (values[0], values[1], values[3])
                        voxelsOnCam[i].append(vCoord)

        not_current_image = (255-currImgs[i])
        nowOffPixels = cv.bitwise_and(not_current_image, mask_xor)
        nowOffPixels = np.argwhere(nowOffPixels == 255)
        for pixel in nowOffPixels:
            x, y = pixel
            if (x, y) in pixels:
                for values in pixels[(x, y)]:
                    if values[3] == i:
                        vCoord = (values[0], values[1], values[3])
                        for j in range(len(voxelsOnCam[i]) - 1):
                            if voxelsOnCam[i][j] == vCoord:
                                del voxelsOnCam[i][j]
                                break

    data, colors = finaliseVoxels(width, height, depth)
    print("My new method took", time.time() - start_time, "to run")
    return data, colors


def cluster(voxels):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    clusteredVoxels = []
    persons = {}

    for voxel in voxels:
        voxel = np.delete(voxel, 1)
        clusteredVoxels.append(np.float32(voxel))

    clusteredVoxels = np.array(clusteredVoxels)
    voxels = np.array(voxels)

    compactness, labels, centers = cv.kmeans(clusteredVoxels, 4, None, criteria, 10, flags=cv.KMEANS_RANDOM_CENTERS)

    for i in range(4):
        persons[i] = voxels[labels.ravel() == i]
        # person1 = voxels[labels.ravel() == 0]
        # person2 = voxels[labels.ravel() == 1]
        # person3 = voxels[labels.ravel() == 2]
        # person4 = voxels[labels.ravel() == 3]

    return centers, persons


def createColorModel(colorModel):

    for person in colorModel:
        bChannel = []
        gChannel = []
        rChannel = []

        for color in colorModel[person]:
            bChannel.append(color[0])
            gChannel.append(color[1])
            rChannel.append(color[2])
        points = np.arange(0, len(colorModel[person]), 1)
        plt.hist(bChannel, points, color='blue', label='blue', density=True)
        plt.hist(gChannel, points, color='green', label='green', density=True)
        plt.hist(rChannel, points, color='red', label='red', density=True)
        plt.ylabel('points')
        title = "person" + str(person)
        plt.title(title)
        plt.show()


def set_voxel_positions(width, height, depth):
    global frameIndex, previousForegroundImages
    foregroundImages = GenerateForeground()

    #if frameIndex == 1:
    data, colors = FirstFrameVoxelPositions(foregroundImages, width, height, depth)

    #else:
    #    data, colors = (XORFrameVoxelPositions(foregroundImages, previousForegroundImages, width, height, depth))
    previousForegroundImages = foregroundImages
    frameIndex += 100

    centers, persons = cluster(data)

    data.clear()
    colors.clear()

    for center in centers:
        center = [center[0]] + [10] + [center[1]]
        data.append(center)
        colors.append((255, 255, 255))

    intrinsicMatrix, dist = calibrate.loadIntrinsics()

    fileName = "camera_extrinsics2.npz"
    with np.load(fileName) as file:
        rotation, translation = [file[j] for j in ['rvec', 'tvec']]

    imageName = "foreground2.png"
    frameName = "frame2.png"
    image = cv.imread(imageName)
    frame = cv.imread(frameName)

    colorModel = {}
    scalar = 100

    for person in persons:
        color = []
        for voxel in persons[person]:
            x = voxel[0]
            y = voxel[1]
            z = voxel[2]

            if 15 > y > 8:
                voxelPoint = (x * scalar,
                              z * scalar,
                              -y * scalar)
                personCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
                fx = int(personCoordinate[0][0][1])
                fy = int(personCoordinate[0][0][0])
                (b, g, r) = frame[fx, fy]
                if int(b) and int(g) and int(r) > 10:
                    color.append((b, g, r))
                    img = cv.circle(image, (int(personCoordinate[0][0][0]), int(personCoordinate[0][0][1])), 1, (int(b), int(g), int(r)), 2)
        cv.imshow('img', img)
        cv.waitKey(500)

        colorModel[person] = color

    createColorModel(colorModel)

    for person in persons:
        for voxel in persons[person]:
            data.append(voxel)
            colors.append((0, 0, 0))

    cv.destroyAllWindows()

    return data, colors

    # start_time = time.time()
    # Xl = -10
    # Xh = 23
    # Yl = -19
    # Yh = 14
    # Zl = 2
    # Zh = -16
    #
    # data, colors = [], []
    #
    # for x in np.arange(Xl, Xh, 0.5):
    #     for y in np.arange(Yl, Yh, 0.5):
    #         for z in np.arange(Zh, Zl, 0.5):
    #             voxelPoint = np.float32((x, y, z)) * tileSize
    #             Xc = voxelPoint[0]
    #             Yc = voxelPoint[1]
    #             Zc = voxelPoint[2]
    #             boolValues = []
    #             for j in range(4):
    #                 print(j)
    #                 print(voxels[Xc, Yc, Zc][j])
    #                 if GetForegroundValue(foregroundImages, j, voxels[Xc, Yc, Zc][j]):
    #                     boolValues.append(True)
    #                     print("done")
    #                 else:
    #                     boolValues.append(False)
    #                     break
    #
    #             # #print(boolValues)
    #             scalar = 0.01
    #             fixedPoint = (Xc * scalar, -Zc * scalar, Yc * scalar)
    #
    #             if np.all(boolValues):
    #                 data.append((fixedPoint[0] + 15,
    #                              fixedPoint[1],
    #                              fixedPoint[2]))
    #                 colors.append([x / width, z / depth, y / height])
    # print("done")
    # print("My old method took", time.time() - start_time, "to run")
    # frameIndex += 10
    # return data, colors


def get_cam_positions():
    # loading lookup table from json file
    global pixels, voxels
    pixels = loadPickle("xor")
    voxels = loadPickle("voxels")
    bs.createBackgroundModel()

    rvecs, tvecs = getData()
    Positions = []
    for i in range(4):
        rotM = cv.Rodrigues(rvecs[i])[0]
        camPos = -rotM.T.dot(tvecs[i])
        camPosFix = [camPos[0], -camPos[2], camPos[1]]
        Positions.append(camPosFix)

    return [Positions[0], Positions[1], Positions[2], Positions[3]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    rvecs, tvecs = getData()
    RotMs = []
    for i in range(4):
        rvec = np.array((rvecs[i][0], rvecs[i][1], rvecs[i][2]))
        rotM = cv.Rodrigues(rvec)[0]
        rotM1 = np.identity(4)
        rotM1[:3, :3] = rotM
        RotMs.append(rotM1)

    cam_angles = [[0, 0, 90], [0, 0, 90], [0, 0, 90], [0, 0, 90]]
    cam_rotations = [glm.mat4(RotMs[0]), glm.mat4(RotMs[1]), glm.mat4(RotMs[2]), glm.mat4(RotMs[3])]

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
