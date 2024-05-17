CXX = pgc++
CXXFLAGS = -O2 -Wall -std=c++20 -fast -acc=gpu -gpu=ccall -Minfo=all
LD = pgc++
LDFLAGS = -fast -acc -gpu=ccall -lstdc++ -lboost_program_options

SOURCES = test.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGETS = gpu

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

all: $(TARGETS)

gpu: $(OBJECTS)
	$(LD) -o $@ $^ $(LDFLAGS) 

clean:
	rm -f $(TARGETS) $(OBJECTS) *.mod core

.PHONY: all clean
