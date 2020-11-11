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
import os
import shutil
import logging
import logging.config

# Logging
logging.config.fileConfig('logger.conf')
log = logging.getLogger()

class Camera:
    def __init__(self,camera_stream=0,calibration_file=None,resolution=None):
        # Create a VideoCapture object
        if str.isdigit(str(camera_stream)) :
            self.CAP = cv.VideoCapture(int(camera_stream),cv.CAP_DSHOW)
        else :
            self.CAP = cv.VideoCapture(str(camera_stream))

        if self.getCapFps() > 0:
            log.info('Camera fps are {0}'.format(self.getCapFps()))
        
        # Set calibration if it has been saved
        if not calibration_file==None :
            with np.load(str(calibration_file)) as X:
                log.info('Calibration file "'+calibration_file+'" loaded')
                self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        
        # Force resolution if specified
        if not resolution==None :
            self.setResolution(resolution[0],resolution[1])

    def __del__(self):
        self.CAP.release()

    def getFrame(self,loop=False):
        """
        getFrame Method get the next frame

        :param loop: Set to true if you are using a video as input and want to loop on it
        :return: frame
        """
        # Capture frame-by-frame
        if self.CAP.isOpened() :
            ret, frame = self.CAP.read()
            if not ret and loop:
                    # End of video reached reset to the first frame
                    self.CAP.set(cv.CAP_PROP_POS_FRAMES, 0)
                    return self.getFrame()

           
        else:
            log.error('Unable to read camera feed.')

        return frame

    @staticmethod  
    def checkKey(key='q'):
        """
        checkKey Method check if key is pressed to break loop

        :param key: keycode to press
        """
        if  cv.waitKey(1) & 0xFF == ord(str(key)):
            cv.destroyAllWindows()
            quit()

    def getCap(self):
        """
        getCap Method get the OpenCV capture object

       :return: OpenCV capture object
        """
        return self.CAP

    def getCapFps(self):
        """
        getCapFps Method get the FPS used by the camera

        :return: fps rate
        """
        return self.CAP.get(cv.CAP_PROP_FPS)

    def getResolution(self):
        return self.CAP.get(cv.CAP_PROP_FRAME_WIDTH),self.CAP.get(cv.CAP_PROP_FRAME_HEIGHT)

    def setResolution(self, x,y):
        self.CAP.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
        self.CAP.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
        x,y=self.getResolution()
        log.info('Camera resolution is '+str(x)+'x'+str(y))

    def getNbFrames(self):
        """
        getNbFrames Method get the number of frames of the capture

        :return: Number of frames
        """
        return int(self.CAP.get(cv.CAP_PROP_FRAME_COUNT))
        
    @staticmethod    
    def recordStream(stream,output=None,fps=20):
        """
        recordStream Method use to record a frame stream

        :param stream: Frames (array of frame or video file)
        :param output: Output path to save the record
        :param fps: Framerate
        :return: output path of the record
        """
        if len(stream) > 0 :
            if output is None:
                output=datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.mp4'
            if os.path.dirname(output) and not os.path.exists(os.path.dirname(output)):
                # directory doesn't exist
                os.makedirs(os.path.dirname(output))
            # Define the codec and create VideoWriter object
            height, width, channels = stream[0].shape
            out = cv.VideoWriter(output,cv.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
            for frame in stream:
                out.write(frame)
            out.release()
            log.info('Record of '+str(len(stream))+' frames successful : '+str(output))    
        else : 
            log.error('Stream is empty')

        return output

    @staticmethod
    def cutStream(stream,output_dir=None,nb_cut=40,frame_start=0.1,frame_end=0.9):
        """
        cutStream Method Take snapshot from a frame stream (array or video file)

        :param stream: Frames (array of frame or video file)
        :param output_dir: Output path to save snapshots
        :param nb_cut: Number of snapshots
        :param frame_start: First frame ([0-1]%)
        :param frame_end: Last frame ([0-1]%)
        """
        # Check frame interval
        if frame_end<frame_start :
            log.error('Frame_start should be < than frame_end')
            quit()

        # Check if directory exist
        if not output_dir :
            output_dir='./'
        
        if os.path.dirname(output_dir) and not os.path.exists(os.path.dirname(output_dir)):
            # directory doesn't exist
            os.makedirs(os.path.dirname(output_dir))

        if isinstance(stream,str):
            # Load video from file
            cam=cv.VideoCapture(stream)
            frame_start=int(cam.get(cv.CAP_PROP_FRAME_COUNT)*frame_start)
            frame_end=int(cam.get(cv.CAP_PROP_FRAME_COUNT)*frame_end)
            interval=int((frame_end-frame_start)/nb_cut)
            log.info('Video loaded "{0}" have {1} frames'.format(stream,str(cam.get(cv.CAP_PROP_FRAME_COUNT))))
            log.info('Cut {0} times, from frame {1} to frame {2}'.format(str(nb_cut),str(frame_start),str(frame_end)))

            # Cut stream
            for i in range(nb_cut):
                ret, frame = cam.read()
                cut=output_dir+'cut_'+str(i)+'.jpg'
                cv.imwrite(cut, frame)
                log.info(cut)
        else:
            # Load from stream array
            frame_start=int(len(stream)*frame_start)
            frame_end=int(len(stream)*frame_end)
            interval=int((frame_end-frame_start)/nb_cut)
            log.info('Stream have {0} frames'.format(len(stream)))
            log.info('{0} times, from frame {1} to frame {2}'.format(str(nb_cut),str(frame_start),str(frame_end)))

            # Cut stream
            for i in range(nb_cut):
                frame=stream[int(frame_start+i*interval)]
                cut=output_dir+'cut_'+str(i)+'.jpg'
                cv.imwrite(cut, frame)
                log.info(cut)

        return output_dir

    @staticmethod
    def createCalibrationFile(jpeg_directory,output='calibration.npz',rows=7,cols=6):
        """
        createCalibrationFile Method create a calibration file

        :param jpeg_directory: Image source directory to make calibration (use .jpg)
        :param output: Output path to save calibration (.npz)
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
            
        log.info('Total error: ', error / len(objpoints))
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
        log.info('Calibration successful')

        return output

    @staticmethod
    def reconstruction(jpeg_directory,calibration_file,rows=7,cols=6):
        """
        reconstruction Method is used to test the camera calibration

        :param jpeg_directory: Image source directory to test calibration (use .jpg)
        :param calibration_file: Path to the calibration (.npz) file
        :param rows: Define the chess board rows 
        :param cols: Define the chess board columns 
        """
        # Load previously saved data
        with np.load(calibration_file) as X:
            mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        
        # reconstruction
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                        [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

        for fname in glob.glob(str(jpeg_directory)+'/*.jpg'):
            img = cv.imread(fname)
            img = cv.resize(img,(512,512))
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (rows,cols),None)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1,2)
                # draw ground floor in green
                img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
                # draw pillars in blue color
                for i,j in zip(range(4),range(4,8)):
                    img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
                # draw top layer in red color
                img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
                cv.imshow('Reconstruction press any key to continue ("s" to save image)',img)
                k = cv.waitKey(0) & 0xFF
                if k == ord('s'):
                    out='calibration-'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.png'
                    cv.imwrite(out, img)
                    log.info(out+'" saved')

        cv.destroyAllWindows()
        
# *** TEST
def test():
    path='./test/'
    # Remove test directory
    if os.path.exists(path):
        shutil.rmtree(path)

    camera=Camera(0) 
    stream=[]
    while True:
        frame=camera.getFrame()
        if frame is not None:
            stream.append(frame)
            cv.imshow('[Camera] stream (press "q" to end capture)',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
        else:
            break

    # [TEST] Record
    record=Camera.recordStream(stream,path+'capture.mp4')
    # [TEST] Cut
    #Camera.cutStream(stream,output_dir=path)
    Camera.cutStream(record,output_dir=path)
    # [TEST] Create Calibration
    camera.createCalibrationFile(path,path+'calibration.npz')
    # [TEST] Reconstruction
    Camera.reconstruction(path,path+'calibration.npz')

#test()