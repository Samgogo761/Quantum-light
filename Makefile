FC      = gfortran
FFLAGS  = -O2 -fopenmp -Wall -Wextra -std=f2008 -fall-intrinsics
LDFLAGS = -llapack -lblas -lfftw3

# Intel oneAPI alternative (uncomment):
# FC      = ifx
# FFLAGS  = -O3 -qopenmp -qmkl -fpp
# LDFLAGS =

SDIR = src
ODIR = obj
MDIR = mod

FC_BASENAME := $(notdir $(FC))
ifneq (,$(filter ifort ifx,$(FC_BASENAME)))
	MODFLAG = -module $(MDIR)
	MODINC  = -I$(MDIR)
else
	MODFLAG = -J$(MDIR)
	MODINC  = -I$(MDIR)
endif

SRCS = mod_params.f90 mod_wannier.f90 mod_crystal.f90 \
       mod_laser.f90 mod_current.f90 mod_quantum_light.f90 \
       mod_sbe.f90 mod_hhg.f90 mod_berry.f90 mod_ensemble.f90 main.f90

OBJS = $(patsubst %.f90,$(ODIR)/%.o,$(SRCS))
PROG = hhg_sbe

.PHONY: all dirs clean debug test

all: dirs $(PROG)

dirs:
	@mkdir -p $(ODIR) $(MDIR)

$(PROG): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

$(ODIR)/%.o: $(SDIR)/%.f90 | dirs
	$(FC) $(FFLAGS) $(MODINC) -c $< -o $@ $(MODFLAG)

debug: FFLAGS = -O0 -g -fbacktrace -fcheck=all -fopenmp -Wall -std=f2008 -fall-intrinsics
debug: clean all

clean:
	rm -rf $(ODIR) $(MDIR) $(PROG) *.mod

# --- Module dependencies ---
$(ODIR)/mod_wannier.o:        $(ODIR)/mod_params.o
$(ODIR)/mod_crystal.o:        $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o
$(ODIR)/mod_laser.o:          $(ODIR)/mod_params.o
$(ODIR)/mod_current.o:        $(ODIR)/mod_params.o
$(ODIR)/mod_quantum_light.o:
$(ODIR)/mod_sbe.o:            $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o $(ODIR)/mod_laser.o
$(ODIR)/mod_hhg.o:            $(ODIR)/mod_params.o
$(ODIR)/mod_berry.o:          $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o $(ODIR)/mod_sbe.o
$(ODIR)/mod_ensemble.o:       $(ODIR)/mod_quantum_light.o $(ODIR)/mod_laser.o \
                              $(ODIR)/mod_sbe.o
$(ODIR)/main.o:               $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o $(ODIR)/mod_laser.o \
                              $(ODIR)/mod_sbe.o $(ODIR)/mod_current.o \
                              $(ODIR)/mod_hhg.o $(ODIR)/mod_quantum_light.o \
                              $(ODIR)/mod_berry.o

# --- Sampler unit test (standalone, no LAPACK/FFTW needed) ---
test: dirs
	$(FC) -O2 -Wall -std=f2008 $(MODINC) $(MODFLAG) -o test_qlight \
		$(SDIR)/mod_quantum_light.f90 tests/test_qlight_sampling.f90
	./test_qlight
