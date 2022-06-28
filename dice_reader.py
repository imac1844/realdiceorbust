import cv2 as cv
import sys
import time
import numpy as np
import imutils
import math
import pytesseract
import copy


pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/TesseractOCR/tesseract.exe'
imgName = str(round(time.time()))
startTime = time.time()

inputImg = './Images/unprocessed/new.jpg'

# Debugging info
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
magenta = (255,0,255)
cyan = (255,255,0)
yellow = (0,255,255)
white = (255,255,255)


class Tray:
###__________Tray init and Attributes__________###
	def __init__(self, inputImg):
		self.name = imgName + "_tray"			#Image name generated from program init time (NOT IN CLASS)
		self.newImgStdSize = (900,1200)
		self.trayAreaBounds = (10000,1000000)
		self.trayStdSize = (600,600)
		self.diceFindSize = (1000,10000)
		self.diceOverlapSearchRadius = 20

		self.image = [0,0,0,0,0,0,0]
		self.image[0] = cv.imread(inputImg)		#self.image() holds all versions of the image in an array
		self.image[1] = self.process_tray_image_grey()
		self.trayBounds = self.contour_rect() 	#boundaries of the tray. I call self.image[1] and am called to define self.image[2], DON'T MOVE ME
		self.image[2] = self.rotate_crop_tray()
		self.image[3], self.image[4] = self.process_tray_image_binary()
		#image types 5 and 6 are defined in method self.get_im5() and self.get_im6() respectively
		self.imageType = [						#The order of self.image() is kept here
			"Original",			#0
			"Grey Uncropped",	#1
			"Grey Cropped",		#2
			"Standard (Grey)",	#3
			"Standard (Binary)",	#4
			"Contours and Boxes (Tray Edge Search)", 	#5
			"Contours and Boxes (Dice Search)" 			#6
		]
		self.diceList = self.find_dice()		#coords of die bounding boxes, in form ([coords], center)
		self.dice = self.crop_dice()			#dice as Dice objects
		# print(self.diceSep)


###__________Tray Classes__________###


	class Dice:
	###__________Dice Classes__________###
		class Image:
		###__________Image init and Attributes__________###
			def __init__(self, img, name):
				self.image = img
				self.name = name
				self.height = img.shape[0]
				self.width = img.shape[1]

		class d4:
			def __init__(self):
				self.sides = 3
				self.values = range(1,5)

		class d6:
			def __init__(self):
				self.sides = 4
				self.values = range(1,7)
				self.reference = self.load_reference()
				self.refcontours = self.load_refcontours()

			def load_reference(self):
				reference = cv.imread("./Images/shape_references/square.jpg")
				greyImg = cv.cvtColor(reference, 6)
				return greyImg

			def load_refcontours(self):
				contours, _ = cv.findContours(self.reference, 1, 2)
				return contours

		class d8:
			def __init__(self):
				self.sides = 6
				self.values = range(1,9)

		class d10:
			def __init__(self):
				self.sides = 6
				self.values = range(1,11)

		class d12:
			def __init__(self):
				self.sides = 10
				self.values = range(1,13)

		class d20:
			def __init__(self):
				self.sides = 6
				self.values = range(1,21)

		class d100:
			pass

	###__________Dice init and Attributes__________###
		def __init__(self, greyImage, center): #center references where the center of the image is in the TRAY, not in the image of the die
			self.name = imgName + str(center)		#Image name generated from program init time
			self.center = center 					#Center of image

			self.images = []
			self.images.append(self.Image(greyImage, self.name + "_Grey"))			#self.image() holds all versions of the image in an array
			self.images.append(self.Image(self.binary_conv(), self.name + "_Binary"))			
			self.images.append([])  #self.image[2] is defined in method self.get_im2(), if called
			self.images.append(self.Image(self.binary_conv(1), self.name + "_Binary_Outline"))
			self.images.append([]) #self.image[4] is defined in method self.get_im4()
			self.imageType = [						#The order of self.images() is kept here
				"Grey",		#0
				"Binary",	#1
				"Contours", 	#2
				"Binary Outline", #3
				"Binary Outline Contours" #4
			]

	###__________Dice Methods__________###
		def find_contours(self, imtype):
			cannyImg = cv.Canny(self.images[imtype].image, 0.1, 0.2, 2)
			contours, _ = cv.findContours(cannyImg, 1, 2)
			self.images[imtype].contours = contours
			return(contours)

		def draw_contours(self, imtype, minlen=0):
			contours = self.find_contours(imtype)
			drawing = np.zeros((self.images[imtype].height, self.images[imtype].width, 3), dtype=np.uint8)
			for i, c in enumerate(contours):
				if len(c) < minlen:
					continue
				cv.drawContours(drawing, contours, i, red)
				cv.circle(drawing, self.center, 0, green, -1)	
			return(drawing)

		def get_im2(self):
			self.images[2] = self.Image(self.draw_contours(0,7), self.name + "_Contours")

		def get_im4(self):
			"""
			Adds an n pixel border to the outside in black to help define lines on the edges
			"""
			#defines the arrays to be added
			addSize = 2
			topBottom = np.zeros((addSize, self.images[3].width), dtype=np.uint8)
			sides = np.zeros((self.images[3].height+2*addSize, addSize), dtype=np.uint8)

			#temporary changes to attributes for the self.draw_contours call
			self.images[4] = self.Image(self.images[3].image, "temp4")

			#adds the border to image3
			self.images[4].image = np.append(topBottom, self.images[4].image, 0)
			self.images[4].image = np.append(self.images[4].image, topBottom, 0)
			self.images[4].image = np.append(sides, self.images[4].image, 1)
			self.images[4].image = np.append(self.images[4].image, sides, 1)

			self.images[4] = self.Image(self.images[4].image, "Binary Outline Contours")
			self.images[4].image = self.draw_contours(4)

		def binary_conv(self, binType = 0):
			#Converts to binary. Returns the processed image.
			if binType == 0:
				binaryImg = (cv.adaptiveThreshold(self.images[0].image,255,cv.ADAPTIVE_THRESH_MEAN_C,\
		        	cv.THRESH_BINARY,11,-15))
			elif binType == 1:
				_, binaryImg = cv.threshold(self.images[0].image, 50, 255,cv.THRESH_BINARY)
			return(binaryImg)

		def show(self, imtype=-1):
			if imtype < 0:
				for i, c in enumerate(self.imageType):
					self.show(i)
			if imtype == 2:
				self.get_im2()
			if imtype == 4:
				self.get_im4()
			if imtype >= 0:
				cv.imshow(self.imageType[imtype] + ", " + str(self.name), self.images[imtype].image)

		def save(self, imtype=-1):
			if imtype < 0:
				for i, c in enumerate(self.imageType):
					self.show(i)
			if imtype == 2:
				self.get_im2()
			if imtype == 4:
				self.get_im4()
			if imtype >= 0:
				cv.imwrite("./Images/processed/" + self.name + "_" + self.imageType[imtype] + ".jpg", self.images[imtype].image)

		def find_sides(self):
			self.get_im4()

			drawing2 = np.zeros((self.images[4].height, self.images[4].width, 3), dtype=np.uint8)

			# self.show(4)
			for i, c in enumerate(self.images[4].contours):
				# print(c)
				hull = cv.convexHull(c)
				print(len(hull))
				cv.drawContours(drawing2, [hull], -1, yellow)
				cv.imshow("ripKobe", drawing2)
				cv.waitKey()			

			# for i, c in enumerate(cnt):
			# 	print(i, c)
			# 	epsilon = 0.1*cv.arcLength(c,True)
			# 	cv.drawContours(drawing2, cv.approxPolyDP(c, epsilon, True), 0, cyan, 2)
			# 	cv.drawContours(drawing2, c, 0, yellow, 2)




###__________Tray Methods__________###
	def show_all_dice(self, imType=-1):
		for d in self.dice:
			d.show(imType)

	def get_im5(self):
		self.image[5] = self.contour_rect(True)

	def get_im6(self):
		self.image[6] = self.find_dice(True)

	def process_tray_image_grey(self):
		#Initial processing	
		sizedImg = cv.resize(self.image[0], self.newImgStdSize, 0, 0)
		greyImg = cv.cvtColor(sizedImg, 6)
		procImg = cv.blur(greyImg, (3,3))
		return (procImg)

	def process_tray_image_binary(self):
		#resizes, converts to binary. returns the processed image.
		sizedImg = cv.resize(self.image[2], self.trayStdSize, 0, 0)
		blurImg = cv.blur(sizedImg, (5,5))
		_, binaryImg = cv.threshold(blurImg, 50, 255,cv.THRESH_BINARY)
		return(sizedImg, binaryImg)

	def rotate_crop_tray(self):
		#Math to calculate how rotated the box is
		ogImgCenter = [(len(self.image[1])/2), ((len(self.image[1][0])/2))]
		bigBoxRotRatio = (self.trayBounds[1][0]-self.trayBounds[0][0])/(self.trayBounds[1][1]-self.trayBounds[0][1])
		bigBoxRotRad = math.atan(bigBoxRotRatio)
		bigBoxRotDeg = math.degrees(bigBoxRotRad)

		#Rotates the opposite amount
		rotated = imutils.rotate_bound(self.image[1], bigBoxRotDeg)
		newImgCenter = [(len(rotated)/2), ((len(rotated[0])/2))]

		#More math, calculates where two corners of the rectangle are after the rotation, then formats them for cropping
		newCorners = []
		for i in 1,3:
			newx = int(((self.trayBounds[i][0]-ogImgCenter[1])*math.cos(bigBoxRotRad))-(self.trayBounds[i][1]-ogImgCenter[0])*math.sin(bigBoxRotRad) + newImgCenter[1])
			newy = int(((self.trayBounds[i][1]-ogImgCenter[0])*math.cos(bigBoxRotRad))+(self.trayBounds[i][0]-ogImgCenter[1])*math.sin(bigBoxRotRad) + newImgCenter[0])
			newCorners.append([newx,newy])
		arrangedCorners = [sorted([newCorners[0][1],newCorners[1][1]]),sorted([newCorners[0][0],newCorners[1][0]])] #[[y,y1],[x,x1]]
		croppedImg = rotated[arrangedCorners[0][0]:arrangedCorners[0][1], arrangedCorners[1][0]:arrangedCorners[1][1]]

		# cv.imshow("cropped", croppedImg)
		# cv.waitKey() 
		return (croppedImg)

	def contour_rect(self, keep5=False):
		"""
		2 USES: if keep5 is False, returns the coords of the largest-in-range rotatedRect. If True, returns the drawing
		"""
		#Canny Edge Detection, contour map
		cannyImg = cv.Canny(self.image[1], 100, 150, 2)
		contours, _ = cv.findContours(cannyImg, 1, 2)

		#minimum area rotated rectangle for each contour
		minRect = [None]*len(contours)
		for i, c in enumerate(contours):
			minRect[i] = cv.minAreaRect(c)

		#Drawing tools init, for debugging
		if keep5:
			drawing = np.zeros((cannyImg.shape[0], cannyImg.shape[1], 3), dtype=np.uint8)

		#variables declared, all empty. These are used for sorting
		pyBox = []
		sortedbox = []
		bigBox = [0,0]
		boxKeep = []

		for i, c in enumerate(contours):
			# Drawings for debugging
			if keep5:
				cv.drawContours(drawing, contours, i, red)

			# Creates a rotated rectangle, not kept between iterations
			box = cv.boxPoints(minRect[i])
			box = np.intp(box)

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

			#saves boxes that meet the criteria for size and ratio
			if 0.9 < abs(dy/(dx+0.0000001)) < 1.1: #+0.0...01 is for 0 catching
				if self.trayAreaBounds[0] < boxArea < self.trayAreaBounds[1]:
					boxKeep.append(box)
					if boxArea > bigBox[1]:
						bigBox = [box, boxArea]
				elif keep5:
					cv.drawContours(drawing, [box], 0, magenta)
			elif keep5:
				cv.drawContours(drawing, [box], 0, cyan)
		if keep5:
			cv.drawContours(drawing, [bigBox[0]], 0, green)
			return(drawing)
		return(bigBox[0])

	def find_dice(self, keep6=False):
		#Canny Edge Detection, contour map
		cannyImg = cv.Canny(self.image[4], 1, 50, 2)
		contours, _ = cv.findContours(cannyImg, 1, 2)

		#minimum area rotated rectangle for each contour
		minRect = [None]*len(contours)
		for i, c in enumerate(contours):
			minRect[i] = cv.minAreaRect(c)

		#Drawing tools init, for debugging
		if keep6:
			drawing = np.zeros((cannyImg.shape[0], cannyImg.shape[1], 3), dtype=np.uint8)

		#variables declared, all empty. These are used for sorting
		pyBox = []
		sortedbox = []
		boxKeep = []
		centersList = []

		for i, c in enumerate(contours):
			# Drawings for debugging
			if keep6:
				cv.drawContours(drawing, contours, i, red)

			# Creates a rotated rectangle, not kept between iterations
			box = cv.boxPoints(minRect[i])
			box = np.intp(box)
			center = (int(minRect[i][0][0]),int(minRect[i][0][1]))
			centersList.append(center)

			#removes duplicates
			inList = False
			for k in range(0, len(centersList)-1):
				if ((centersList[k][0] - center[0])**2 + (centersList[k][1] - center[1])**2) < self.diceOverlapSearchRadius**2:
					inList = True
					del centersList[-1]
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
			if 0.8 < abs(dy/(dx+0.01)) < 1.3: #+0.01 is for 0 catching
				if self.diceFindSize[0] < boxArea < self.diceFindSize[1]:
					if not inList:
						boxKeep.append([box, center])
						if keep6:
							cv.drawContours(drawing, [box], 0, green)
							cv.circle(drawing, center, 0, green, -1)	
				elif keep6:
					pass
					cv.drawContours(drawing, [box], 0, magenta)
			elif keep6:
				pass
				cv.drawContours(drawing, [box], 0, cyan)
		if keep6:
			return(drawing)
		return(boxKeep)

	def crop_dice(self):
		croppedDice = []
		ogImgCenter = [self.trayStdSize[0]/2, self.trayStdSize[1]/2]
		allDice = []

		for i, (coords, center) in enumerate(self.diceList):
			dieRotRatio = (coords[1][0]-coords[0][0])/(coords[1][1]-coords[0][1]+0.000001)
			dieRotRad = math.atan(dieRotRatio)
			dieRotDeg = math.degrees(dieRotRad)
			die = imutils.rotate_bound(self.image[3], dieRotDeg)
			newImgCenter = [(len(die)/2), ((len(die[0])/2))]
			
			newCorners = []
			for j in 1,3:
				newx = int(((coords[j][0]-ogImgCenter[0])*math.cos(dieRotRad)) - (coords[j][1]-ogImgCenter[1])*math.sin(dieRotRad) + newImgCenter[0])
				newy = int(((coords[j][1]-ogImgCenter[1])*math.cos(dieRotRad)) + (coords[j][0]-ogImgCenter[0])*math.sin(dieRotRad) + newImgCenter[1])
				newCorners.append([newx,newy])

			arrangedCorners = [sorted([newCorners[0][1],newCorners[1][1]]),sorted([newCorners[0][0],newCorners[1][0]])] #[[y,y1],[x,x1]]
			croppedDie = die[arrangedCorners[0][0]:arrangedCorners[0][1], arrangedCorners[1][0]:arrangedCorners[1][1]]

			allDice.append(self.Dice(croppedDie, center))
			# cv.imshow("Die at "+ str(center), croppedDie)
			# cv.waitKey()

		return (allDice)

	def show(self, imtype=-1):
		if imtype < 0:
			for i, c in enumerate(self.imageType):
				self.show(i)
		elif imtype == 5:
			self.get_im5()
		elif imtype == 6:
			self.get_im6()
		if imtype >= 0:
			cv.imshow(self.imageType[imtype] + ", " + str(self.name), self.image[imtype])

	def save(self, imtype=-1):
		if imtype < 0:
			for i, c in enumerate(self.imageType):
				self.save(i)
		elif imtype == 5:
			self.get_im5()
		elif imtype == 6:
			self.get_im6() 
		if imtype >= 0:
			cv.imwrite("./Images/processed/" + self.name + "_" + self.imageType[imtype] + ".jpg", self.image[imtype])

i=0

tray = Tray(inputImg)
die = tray.dice[i].images[3].image
ref_square = tray.Dice.d6().reference

ret = cv.matchShapes(die,ref_square,1,0.0)

#tray.dice[i].show()
print(ret)
print(ret < 0.001)


# i = 9


# tray.dice[i].show(4)
# tray.dice[i].find_sides()



# img1 = cv.imread('star.jpg',0)
# img2 = cv.imread('star2.jpg',0)


# ret, thresh = cv.threshold(img1, 127, 255,0)
# ret, thresh2 = cv.threshold(img2, 127, 255,0)
# im2,contours,hierarchy = cv.findContours(thresh,2,1)
# cnt1 = contours[0]
# im2,contours,hierarchy = cv.findContours(thresh2,2,1)
# cnt2 = contours[0]

# tray.dice[0].find_sides()
# print(hull)






# print(pytesseract.image_to_string(r'D:/Desktop/Ian/Coding/dice_connection/Images/test2BW.jpg')))











endTime = round(time.time() - startTime, 5)
print(2*"\n", len(tray.dice), "dice in image, processing took ", endTime)

cv.waitKey()



####				!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!							matchShapes()