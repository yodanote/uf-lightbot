CC=g++

OPENCV_FLAGS=$(shell pkg-config --cflags opencv)
OPENCV_LIBS=$(shell pkg-config --libs opencv)

all: 
	$(CC) HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS)

cam_test: CamFaceDetect.cpp
	$(CC) CamFaceDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -ocam_test

lip_test: LipFeatureDetect.cpp HaarCascadeObjectDetector.cpp
	$(CC) LipFeatureDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -olip_test

lip_contour: LipContour.cpp HaarCascadeObjectDetector.cpp
	$(CC) LipContour.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -olip_contour

eye_detect: EyeDetect.cpp HaarCascadeObjectDetector.cpp
	$(CC) EyeDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -oeye_test

