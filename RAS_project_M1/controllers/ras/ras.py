from rasrobot import RASRobot

import numpy as np
import time

import cv2
#from cv2 import dnn_superres

class MyRobot(RASRobot):
    
    def __init__(self):

        super(MyRobot, self).__init__()
        # Initialise and resize a new window 
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 128*4, 64*4)

    #Takes an image with detected edges as input and calculates the steering angle to keep the robot on the road.
    def get_road(self, edges):
        
        #Follow the yellow line by turning left or right depending on where the line is in the image.
       
        # Crop the image to only look at the bottom half
        height, width = edges.shape
        cropped = edges[3*height//4:height,:]
        
        # Get the indices of the white pixels
        indices = np.where(cropped == 255)
        
        # Check if there are any white pixels in the image
        if len(indices[0]) == 0:
            return 0
        
        # Compute the leftmost white pixel
        leftmost = np.min(indices[1])
        #print(leftmost)
        # Compute the deviation from the left edge of the image
        deviation = leftmost - width/4
        
        # Threshold the deviation to stay within the lane
        if deviation > width/4:
            deviation = width/4
        elif deviation < -width/4:
            deviation = -width/4
        
        # Compute the steering angle
        steering_angle = deviation/(width/2)
        #print(steering_angle)
        return steering_angle
    
    #Takes an image with detected edges as input.
    #Crops the image to only look at the bottom quarter.
    #Finds the indices of the white pixels (edges).
    #Computes the center of the white pixels and calculates the deviation from the center of the image.
    #Computes the steering angle based on the deviation.    
    #yellow lane detection on road
    def yellowline(self, edges):
        
        #Follow the yellow line by turning left or right depending on where the line is in the image.
        
        # Crop the image to only look at the bottom quarter
        # restricting the view since zebra cross is in yello
        height, width = edges.shape
        cropped = edges[3*height//4:height,:]
        # Get the indices of the white pixels
        indices = np.where(cropped == 255)
        
        # Check if there are any white pixels in the image
        if len(indices[0]) == 0:
            return 0
        # Compute the center of the white pixels
        center = np.mean(indices[1])
        #print(center)
        # Compute the deviation from the center of the image
        deviation = center - width/2
        #print(deviation)
        # Compute the steering angle
        steering_angle = deviation/(width/2)
        return steering_angle    
    
    #Contains the main loop of the robot, which executes while self.tick() returns True.
    #In each iteration:
    #Get the camera image and preprocess it (resize, apply CLAHE, and convert to HSV color space).
    #Create a binary mask to filter out yellow color in the image.
    #Apply Gaussian blur and Canny edge detection to the mask.
    #Calculate the steering angle using the yellowline method.
    #Update the robot's speed and steering angle based on the calculated steering angle.
    #Display the output image with the detected edges in the "output" window.
    def run(self):
        
        #This function implements the main loop of the robot.
        
        while self.tick():
            # Get the camera image and convert it to grayscale
            image = self.get_camera_image()
            image = cv2.resize(image, (600,600))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert to LAB color space
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # create a CLAHE object
            image[:,:,0] = clahe.apply(image[:,:,0])  # apply CLAHE to the L channel
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)  # convert back to BGR color space
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            yellowLL = np.array([20, 60, 100])
            #upper_yellow = np.array([35, 167, 166])
            #yellowUL = np.array([36, 160, 162])
            yellowUL = np.array([30, 130, 200])
            mask_yellow = cv2.inRange(hsv, yellowLL, yellowUL)
            #print(mask_yellow)
            # Apply Gaussian blur to mask
            mask_blur = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
            #print(mask_blur)
            # Apply Canny edge detection to mask
            edges = cv2.Canny(mask_blur, 100, 200)

            steering_angle = self.yellowline(edges)
            # If the yellow line ends, turn right
            if steering_angle == 0:
                steering_angle = 0.4
                speed = 30 
                #default_angle=steering_angle
             # thresholding the steering angle to avoid drift
            if steering_angle < -0.3:
                steering_angle =- 0.3
                speed = 40 
                #default_angle=steering_angle
            elif steering_angle > 0.3:
                steering_angle = 0.3
                speed = 40 
                #default_angle=steering_angle
            else:
                 steering_angle=0
                 speed = 40 
                
                       
            # Set the speed and steering angle of the robot
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)
            # Display the output image with the detected edges
            output = np.dstack((edges, edges, edges))
            cv2.imshow('output', output)
            cv2.waitKey(1)

   
              

# The API of the MyRobot class, is extremely simple, not much to explain.
# We just create an instance and let it do its job.
robot = MyRobot()
robot.run()


