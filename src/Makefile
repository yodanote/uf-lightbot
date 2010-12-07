CC=g++

OPENCV_FLAGS=$(shell pkg-config --cflags opencv)
OPENCV_LIBS=$(shell pkg-config --libs opencv)

all: 
	$(CC) HaarCascadeObjectDetector.cpp $(OPENCV_FLAGS) $(OPENCV_LIBS)


