# Makefile for CLL113 Term Project C++ Numerical Solver
CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

TARGET = solver
SRC = main_solver.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o
