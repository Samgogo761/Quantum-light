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
       mod_spin.f90 mod_sbe.f90 mod_hhg.f90 mod_berry.f90 mod_geometry.f90 \
       mod_ensemble.f90 main.f90

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
	rm -rf $(ODIR) $(MDIR) $(PROG) *.mod tests/*.mod test_qlight test_parse_ids

# --- Module dependencies ---
$(ODIR)/mod_wannier.o:        $(ODIR)/mod_params.o
$(ODIR)/mod_crystal.o:        $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o
$(ODIR)/mod_laser.o:          $(ODIR)/mod_params.o
$(ODIR)/mod_current.o:        $(ODIR)/mod_params.o
$(ODIR)/mod_quantum_light.o:
$(ODIR)/mod_spin.o:           $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o
$(ODIR)/mod_sbe.o:            $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o $(ODIR)/mod_laser.o \
                              $(ODIR)/mod_spin.o
$(ODIR)/mod_hhg.o:            $(ODIR)/mod_params.o
$(ODIR)/mod_berry.o:          $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o $(ODIR)/mod_sbe.o
$(ODIR)/mod_geometry.o:       $(ODIR)/mod_params.o $(ODIR)/mod_crystal.o \
                              $(ODIR)/mod_sbe.o
$(ODIR)/mod_ensemble.o:       $(ODIR)/mod_quantum_light.o $(ODIR)/mod_laser.o \
                              $(ODIR)/mod_sbe.o
$(ODIR)/main.o:               $(ODIR)/mod_params.o $(ODIR)/mod_wannier.o \
                              $(ODIR)/mod_crystal.o $(ODIR)/mod_laser.o \
                              $(ODIR)/mod_sbe.o $(ODIR)/mod_current.o \
                              $(ODIR)/mod_hhg.o $(ODIR)/mod_quantum_light.o \
                              $(ODIR)/mod_berry.o $(ODIR)/mod_geometry.o \
                              $(ODIR)/mod_spin.o

# --- Sampler + propagate_ids unit tests (standalone, no LAPACK/FFTW needed) ---
# NOTE: remove tests/*.mod first — stale local .mod shadows -Imod/-Jmod.
test: dirs
	rm -f tests/*.mod
	$(FC) -O2 -Wall -std=f2008 $(MODINC) $(MODFLAG) -o test_qlight \
		$(SDIR)/mod_quantum_light.f90 tests/test_qlight_sampling.f90
	./test_qlight
	rm -f tests/*.mod
	$(FC) -O2 -Wall -std=f2008 $(MODINC) $(MODFLAG) -o test_parse_ids \
		$(SDIR)/mod_quantum_light.f90 tests/test_parse_propagate_ids.f90
	./test_parse_ids
	python tools/analysis/test_parse_propagate_ids_neg.py --bin ./test_parse_ids
	python tools/analysis/test_hhg_fft_utils.py
	python tools/analysis/test_merge_compare_neg.py
	python tools/analysis/test_merge_compare_pos.py
	python tools/analysis/test_gh3_gh5_quadrature_gate.py
	python tools/analysis/test_diagnose_gh5_quadrature.py
	python tools/analysis/test_cep_pi_gate_v2.py
	python tools/analysis/test_gh7_tail_model_gate.py
	$(MAKE) test-field-cep

# Production generate_field_sample antipode gate (mod_params + mod_laser only).
.PHONY: test-field-cep
test-field-cep: dirs
	rm -f tests/*.mod
	$(FC) $(FFLAGS) $(MODINC) $(MODFLAG) -o test_field_cep_pi \
		$(SDIR)/mod_params.f90 $(SDIR)/mod_laser.f90 tests/test_field_cep_pi.f90
	./test_field_cep_pi
	rm -f tests/*.mod
