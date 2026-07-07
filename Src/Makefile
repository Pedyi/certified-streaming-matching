# Makefile — build all binaries for the certified-streaming-matching artifact.
#   make            # build all
#   make test       # quick smoke test (seconds)
#   make clean
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall
BINS = assadi_mbm phase2 phase4 robustness

all: $(BINS)
assadi_mbm: assadi_mbm.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
phase2: phase2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
phase4: phase4.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
robustness: robustness.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
test: robustness
	./robustness --seeds 2 --maxit 30
clean:
	rm -f $(BINS) *.o
.PHONY: all test clean
