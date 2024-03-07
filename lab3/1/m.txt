CC = g++
CFLAGS = -std=c++11 -fopenmp

all: program

program: main.o
	$(CC) $(CFLAGS) main.o -o program

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm -rf *.o program