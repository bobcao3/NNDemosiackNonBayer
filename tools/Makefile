CXX=g++
CCLD=g++
CXXFLAGS=-Wextra -O0 -g

all: image2bayer

image2bayer: lodepng.o image2bayer.cpp filters_definition.hpp image2bayer.hpp
	$(CXX) $(CXXFLAGS) -c image2bayer.cpp -o image2bayer.o
	$(CCLD) image2bayer.o lodepng.o -o image2bayer

lodepng.o: lodepng/lodepng.cpp lodepng/lodepng.h
	$(CXX) $(CXXFLAGS) -c lodepng/lodepng.cpp -Ilodepng -o lodepng.o

clean:
	rm -f *.o
	rm -f image2bayer