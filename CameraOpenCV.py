#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CameraOpenCV.py: Object to manage OpenCV camera"""
__author__      = 'Vincent Berthet'
__license__     = 'MIT'
__email__       = 'vincent.berthet42@gmail.com'
__website__     = 'https://realvincentberthet.github.io/vberthet/'

import cv2 as cv
import numpy as np
import glob
from datetime import datetime

class Camera:
    def __init__(self,camera_stream=0,calibration_file=None):
        # Create a VideoCapture object
        if str.isdigit(str(camera_stream)) :
            self.CAP = cv.VideoCapture(int(camera_stream),cv.CAP_DSHOW)
        else :
            self.CAP = cv.VideoCapture(str(camera_stream))

        # Set calibration if it has been saved
        if not calibration_file==None :
            with np.load(str(calibration_file)) as X:
                self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    def __del__(self):
        self.CAP.release()

    def getFrame(self,loop=False):
        # Capture frame-by-frame
        if self.CAP.isOpened() :
            ret, frame = self.CAP.read()
            if not ret and loop:
                    # End of video reached reset to the first frame
                    self.CAP.set(cv.CAP_PROP_POS_FRAMES, 0)
                    return self.getFrame()
        else:
            print('[ERROR] Unable to read camera feed.')
        
        return frame
        
    def checkKey(self,key='q'):
        if  cv.waitKey(1) & 0xFF == ord(str(key)):
            cv.destroyAllWindows()
            quit()


    def getCap(self):
        return self.CAP

    def getCapFps(self):
            return self.CAP.get(cv.CAP_PROP_FPS)

    def getNbFrames(self):
        return int(self.CAP.get(cv.CAP_PROP_FRAME_COUNT))
        
    @staticmethod    
    def recordStream(stream,output='datetime.mp4',fps=10):
        if len(stream) > 0 :
            if output=='datetime.mp4':
                output=datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.mp4'
            # Define the codec and create VideoWriter object
            height, width, channels = stream[0].shape
            out = cv.VideoWriter(output,cv.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
            for frame in stream:
                out.write(frame)
            out.release()
            print('[INFO] Record of '+str(len(stream))+' frames successful : '+str(output))    
        else : 
            print('[ERROR] Stream is empty')

    @staticmethod
    def createCalibrationFile(jpeg_directory,output='calibration.npz',rows=7,cols=6):
        """
        :param rows: Define the chess board rows 
        :param cols: Define the chess board columns 
        """
        # Set the termination criteria for the corner sub-pixel algorithm
        criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
        # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
        # Create the arrays to store the object points and the image points
        objpoints = []
        imgpoints = []
        images = glob.glob(str(jpeg_directory)+'/*.jpg')
        for fname in images:
            # Load the image and convert it to gray scale
            img = cv.imread(fname)
            img = cv.resize(img,(640,480))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (rows,cols), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (rows,cols), corners2, ret)
                cv.imshow('chess', img)
                cv.waitKey(500)
        cv.destroyAllWindows()

        # Calibrate the camera and save the results
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez(str(output), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        error = 0

        for i in range(len(objpoints)):
            point, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error += cv.norm(imgpoints[i], point, cv.NORM_L2) / len(point)

        print("Total error: ", error / len(objpoints))

        # Load one of the test images
        img2 = cv.imread(images[0])
        img2 = cv.resize(img2,(img.shape[1],img.shape[0]))
        h, w = img.shape[:2]

        # Obtain the new camera matrix and undistort the image
        newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistortedImg = cv.undistort(img2, mtx, dist, None, newCameraMtx)
        # x, y, w, h = roi
        # undistortedImg = undistortedImg[y:y+h, x:x+w]
        # cv.imwrite('calibresult.png', undistortedImg)

        # Display the final result
        cv.imshow('chess', np.hstack((img2, undistortedImg)))
        cv.waitKey(0)
        print('Calibration successful')

    def reconstruction(self):
        pass

    @staticmethod
    def cutVideo():
        pass   



def testRecord():
    camera=Camera('sample.mp4') 
    stream=[]
    while True:
        frame=camera.getFrame()

        if frame.all() :
            break
        
        stream.append(frame)
        cv.imshow('[testRecord] stream',frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

    Camera.recordStream(stream,fps=camera.getCapFps())

# TESTS
# testRecord()
