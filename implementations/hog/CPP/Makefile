# location of the Python header files
 
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
 
# location of the Boost Python include files and library
 
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib
 
# compile mesh classes
#TARGET = hello_ext
 
hog.so: hog.o
	g++ -std=c++14 -O3 -shared -Wl,--export-dynamic hog.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o hog.so -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lm
	sudo cp hog.so /usr/local/lib/python2.7/dist-packages/hog.so
 
hog.o: hog.cpp hog.hpp
	g++ -std=c++14 -O3 -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c hog.cpp -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lm

hog_no_boost.so: hog_no_boost.o
	g++ -std=c++14 -g -shared -Wl,--export-dynamic hog_no_boost.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o hog_no_boost.so -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lopencv_highgui -lm

hog_no_boost.o: hog_no_boost.cpp hog_no_boost.h
	g++ -std=c++14 -g -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c hog_no_boost.cpp -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lm

runHog: hog.cpp hog.hpp runHog.cpp
	clang++ -std=c++14 -g -o runHog runHog.cpp hog.cpp -lopencv_imgcodecs -lopencv_core -L/usr/lib64 -lstdc++ -lopencv_photo -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lm

test.so: test.o
	g++ -std=c++14 -g -shared -Wl,--export-dynamic test.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -I$(PYTHON_INCLUDE) -o test.so

test.o: test.cpp
	g++ -std=c++14 -g -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -Wno-cpp test.cpp

clean:
	rm runHog hog.o hog.so
