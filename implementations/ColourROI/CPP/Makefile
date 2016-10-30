# location of the Python header files
 
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
 
# location of the Boost Python include files and library
 
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib
 
# compile mesh classes
#TARGET = hello_ext
 
colourROI.so: colourROI.o
	g++ -std=c++14 -O3 -shared -Wl,--export-dynamic colourROI.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -I$(PYTHON_INCLUDE) -o colourROI.so -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lm
	sudo cp colourROI.so /usr/local/lib/python2.7/dist-packages/colourROI.so
 
colourROI.o: ROIFindColour.cpp
	g++ -std=c++14 -O3 -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -Wno-cpp ROIFindColour.cpp -o colourROI.o -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lm

clean:
	rm colourROI.o colourROI.so
