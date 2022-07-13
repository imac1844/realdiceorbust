

"""
Pseudo code and general idea

Maybe try doing it for a simple coin?


You are given raw input of a camera, and must calibrate yourself to light. There should be a function to test how well dice will work. 
You should improve as time goes on, learning specific dice. perhapse use statistics to detect mistakes.


raw image -> find tray-> Find dice       //use contrast filter, try and sort them by their shadow. statistics, % this is the same image. Use higher resolution images, if possible.
sort dice, {{d6}{d4}{d12,d20}{d10/d100,d8}}

Sorting is done by high contrast comparisons to sorting filter, meant to break down the task of the number reader

user calibration of dice? Need clean reference images of each die, preferably each face. 

d4/d6:
	contrast filter, trying to match an image and rotation  //maybe have a reference image in each idrection?









Important to work on:

 - self.diceOverlapSearchRadius and self.diceFindSize in DiceConnector __init__ are manually assigned and should be dynamic

"""
import time
import numpy as np
import cv2
import imutils
import math
from threading import Thread

cap_button = 's' #This letter gets encoded to ascii, so it is case sensitive

# Debugging info
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
magenta = (255,0,255)
cyan = (255,255,0)
yellow = (0,255,255)
white = (255,255,255)




class DiceConnector:

	def __init__(self, Capture, frame_i, ThreshTest=False, ShowSteps=False):
		self.threshold_a = 0
		self.wrk_img = 0
		self.Capture = Capture
		self.ThreshTest = ThreshTest
		self.defaultThresh = 230
		self.max_dice = 5

		self.diceFindSize = (8000, 30000)
		self.diceOverlapSearchRadius = 3

		self.show = ShowSteps # Set True to show working images

		self.diceList = self.image_processor()		#coords of die bounding boxes, in form ([coords], center). 
		self.dice = self.list_dice()				#dice as Dice objects
		self.display()

	def list_dice(self):
		croppedDice = []
		ogImgCenter = self.Capture.centerPoint
		allDice = []

		for i, (coords, center) in enumerate(self.diceList):
			dieRotRatio = (coords[1][0]-coords[0][0])/(coords[1][1]-coords[0][1]+0.000001)
			dieRotRad = math.atan(dieRotRatio)
			dieRotDeg = math.degrees(dieRotRad)
			die = imutils.rotate_bound(self.Capture.img, dieRotDeg)
			newImgCenter = [(len(die)/2), ((len(die[0])/2))]
			
			newCorners = []
			for j in 1,3:
				newx = int(((coords[j][0]-ogImgCenter[0])*math.cos(dieRotRad)) - (coords[j][1]-ogImgCenter[1])*math.sin(dieRotRad) + newImgCenter[1])
				newy = int(((coords[j][1]-ogImgCenter[1])*math.cos(dieRotRad)) + (coords[j][0]-ogImgCenter[0])*math.sin(dieRotRad) + newImgCenter[0])
				newCorners.append([newx,newy])

			arrangedCorners = [sorted([newCorners[0][1],newCorners[1][1]]),sorted([newCorners[0][0],newCorners[1][0]])] #[[y,y1],[x,x1]]
			croppedDie = die[arrangedCorners[0][0]:arrangedCorners[0][1], arrangedCorners[1][0]:arrangedCorners[1][1]]

			allDice.append(Dice(croppedDie, center))
		return (allDice)

	def on_change(self, value=100):
		image_copy = self.wrk_img.copy()
		self.threshold_a = value
		_, image_copy = cv2.threshold(image_copy, self.threshold_a, 255, cv2.THRESH_BINARY)
		cv2.imshow('Threshold Test(a)', image_copy)

	def image_processor(self):
		#Process image to a Binary Threshold
		raw_img = self.Capture.img
		self.wrk_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
		self.wrk_img = cv2.GaussianBlur(self.wrk_img, (11,11), 1000)

		if self.show:
			cv2.imshow("1 - Blurred", self.wrk_img)
			cv2.moveWindow("1 - Blurred", -1275,0)

		
		if self.ThreshTest:
			cv2.imshow('Threshold Test(a)', self.wrk_img)
			a = cv2.createTrackbar("Thresh a", "Threshold Test(a)", 0, 255, self.on_change)
			cv2.waitKey(0)
		else:
			self.threshold_a = self.defaultThresh
		_, self.wrk_img = cv2.threshold(self.wrk_img, self.threshold_a, 255, cv2.THRESH_BINARY)


		if self.show:
			cv2.imshow("2 -Threshold", self.wrk_img)
			cv2.moveWindow("2 -Threshold", -1275,515)


		canny_img = cv2.Canny(self.wrk_img, 1, 50, 2)
		contours, _ = cv2.findContours(canny_img, 1, 2)

		#minimum area rotated rectangle for each contour
		minRect = [None]*len(contours)
		for i, c in enumerate(contours):
			minRect[i] = cv2.minAreaRect(c)

		#Drawing tools init, for debugging
		if self.show:
			drawing = np.zeros((canny_img.shape[0], canny_img.shape[1], 3), dtype=np.uint8)


		#variables declared, all empty. These are used for sorting
		pyBox = []
		sortedbox = []
		boxKeep = []
		centersList = []

		SortedContours = sorted(contours, key=cv2.contourArea, reverse=True)

		for i, c in enumerate(SortedContours):
			# Drawings for debugging
			if self.show:
				cv2.drawContours(drawing, contours, i, red)


			# Creates a rotated rectangle, not kept between iterations
			box = cv2.boxPoints(minRect[i])
			box = np.intp(box)
			center = (int(minRect[i][0][0]),int(minRect[i][0][1]))
			centersList.append(center)
			

			#removes duplicates
			inList = False
			for k in range(0, len(centersList)-1):
				if ((centersList[k][0] - center[0])**2 + (centersList[k][1] - center[1])**2) < self.diceOverlapSearchRadius**2:
					inList = True
					centersList.pop()
					break

			#sorts a copy of box, and keeps the sorted boxes in a list
			pyBox.append([])
			for j in range(0,4):
				pyBox[i].append([])
				pyBox[i][j].append(box[j][0])
				pyBox[i][j].append(box[j][1])
			sortedbox.append(sorted(pyBox[i]))

			#determines size and ratio of sides for each box
			dx = sortedbox[i][2][0] - sortedbox[i][0][0]
			dy = sortedbox[i][1][1] - sortedbox[i][0][1]
			boxArea = abs(dx*dy)


			#saves (non duplicate) boxes that meet the criteria for size and ratio
			if 0.5 < abs(dy/(dx+0.01)) < 2: #+0.01 is for 0 catching
				if self.diceFindSize[0] < boxArea < self.diceFindSize[1]:
					if not inList:

						boxKeep.append([box, center])
						if self.show:
							cv2.drawContours(drawing, [box], 0, green)
							cv2.circle(drawing, center, 0, green, -1)	

				elif self.show:
					# cv2.drawContours(drawing, [box], 0, magenta)
					pass
			elif self.show:
				# cv2.drawContours(drawing, [box], 0, cyan)
				pass
		if self.show:
			cv2.imshow("3 - Contours and boxes", drawing)
			cv2.moveWindow("3 - Contours and boxes", -1935,515)
		return(boxKeep)

	def display(self):
		for die, center in (self.diceList):
			cv2.drawContours(self.Capture.img, [die], -1, green, 3)
		cv2.imshow("0 - Live Feed", self.Capture.img)
		cv2.moveWindow("0 - Live Feed", -1935,0)

		if self.show:
			displacement = 0
			for i, d in enumerate(self.dice):
				cv2.imshow("Die "+str(i), d.img)
				cv2.moveWindow("Die "+ str(i), -650,(displacement+(i*30)))
				displacement += d.frameHeight

			for j in range(len(self.dice), self.max_dice):
				try:
					cv2.destroyWindow("Die "+ str(j))
				except cv2.error:
					pass

class Dice:
	def __init__(self, raw_img, center):
		self.center = center
		self.img = raw_img
		self.frameWidth = len(self.img[0])
		self.frameHeight = len(self.img)


class VideoCapture:

		def __init__(self):
			self.frameWidth = 640
			self.frameHeight = 480
			self.centerPoint = (self.frameWidth/2, self.frameHeight/2)
			self.capture = cv2.VideoCapture(1)
			self.capture.set(3, self.frameWidth)
			self.capture.set(4, self.frameHeight)
			self.img = 0 #Defined in the "display" method every frame



		def display(self):
			print("Display loading... (press q to quit) \n")
			frame = 0
			while True:
				# time.sleep(1)
				frame += 1
				_, img = self.capture.read()
				self.img = img
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				# if cv2.waitKey(1) & 0xFF == ord('s'):
				# 	pass

				DiceConnector(self, frame)
				

			self.capture.release()
			cv2.destroyAllWindows()




Webcam = VideoCapture()
display_window = Thread(target=Webcam.display())
display_window.start()
# Webcam.display()



# dice_connection = DiceConnector(Webcam)

# display_window.join()
cv2.waitKey(0)
cv2.destroyAllWindows()