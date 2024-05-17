INFO = -Minfo=all
LIBS = -lboost_program_options
HOST = -acc=host
MULT = -acc=multicore
CXX = pgc++

all:single multi
	

single:
	$(CXX) $(HOST) $(INFO) $(LIBS) -o $@ main.cpp

multi:
	$(CXX) $(MULT) $(INFO) $(LIBS) -o $@ main.cpp

clean:all
	rm serial single
