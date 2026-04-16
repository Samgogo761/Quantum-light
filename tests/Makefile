# Stand-alone unit test for mod_quantum_light.
#
# Builds only the sampler module + test driver, so it compiles even when
# the main SBE solver tree (mod_laser / mod_sbes / MKL FFT) is unavailable.
#
# Usage:
#   cd tests/
#   make              # builds ./test_qlight_sampling
#   ./test_qlight_sampling
#   make clean

FC      ?= gfortran
FFLAGS  ?= -O2 -Wall -Wextra -std=f2008 -fimplicit-none
SRC     := ../src/mod_quantum_light.f90 test_qlight_sampling.f90
BIN     := test_qlight_sampling

.PHONY: all clean run
all: $(BIN)

$(BIN): $(SRC)
	$(FC) $(FFLAGS) -o $@ $(SRC)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) *.mod *.o
