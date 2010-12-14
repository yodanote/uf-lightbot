CC=g++

OPENCV_FLAGS=$(shell pkg-config --cflags opencv)
OPENCV_LIBS=$(shell pkg-config --libs opencv) -lfann

SOURCE=main.cpp FaceDetector.cpp MouthFeatureDetector.cpp NoseFeatureDetector.cpp HaarCascadeObjectDetector.cpp EyeFeatureDetector.cpp BrowFeatureDetector.cpp
HEADERS=FaceDetector.h MouthFeatureDetector.h NoseFeatureDetector.h HaarCascadeObjectDetector.h EyeFeatureDetector.h

all: $(SOURCE) $(HEADERS)
	$(CC) -g $(SOURCE) $(OPENCV_FLAGS) $(OPENCV_LIBS) -olightbot

features: ManualFeatures.cpp
	$(CC) ManualFeatures.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -ofeatures

cam_test: CamFaceDetect.cpp
	$(CC) CamFaceDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -ocam_test

lip_test: LipFeatureDetect.cpp HaarCascadeObjectDetector.cpp
	$(CC) LipFeatureDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -olip_test

lip_contour: LipContour.cpp HaarCascadeObjectDetector.cpp
	$(CC) LipContour.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -olip_contour

eye_detect: EyeDetect.cpp HaarCascadeObjectDetector.cpp
	$(CC) EyeDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -oeye_test

brow_detect: BrowDetect.cpp HaarCascadeObjectDetector.cpp
	$(CC) EyeDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -oeye_test



nose_detect: NoseDetect.cpp HaarCascadeObjectDetector.cpp
	$(CC) NoseDetect.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -onose_test

test:
	$(CC) FaceDetector.cpp HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS) -otest
