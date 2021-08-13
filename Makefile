# Makefile for deep learning scoring (later using Hybrid Monte Carlo)


CC = clang
CXX = clang
MAKE = make
MKDIR = mkdir
AR = ar
CD = cd
RM = rm -f
MV = mv
CP = cp


## Enable this for Windows unicode:
## WINDOWSFLAGS=-municode

CFLAGS = -fPIC -O3 -mtune=native -ffast-math -g -fopenmp -Wno-deprecated `pkg-config --cflags dinrhiw` -I/usr/local/include/ -g -fpermissive $(WINDOWSFLAGS)

CXXFLAGS = -fPIC -O3 -mtune=native -ffast-math -g -fopenmp -Wno-deprecated  `pkg-config --cflags dinrhiw` -I/usr/local/include/ -Wno-attributes -g -fpermissive $(WINDOWSFLAGS)

TARGET=scoring
OBJECTS=main.o

LINUXLIBS=-L/usr/local/lib/ -ldl -lpthread `pkg-config dinrhiw --libs` -lstdc++fs
WINLIBS=-L/usr/local/lib/ -lpthread -lgmp `pkg-config dinrhiw --libs` `pkg-config sdl2 --libs` -lstdc++fs
##LIBS=$(WINLIBS)
LIBS=$(LINUXLIBS)


$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -Wl,-E -o $(TARGET) $(OBJECTS) $(LIBS)
