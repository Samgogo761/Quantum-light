!===============================================================================
! PATCH SNIPPET for src/main.f90
!-------------------------------------------------------------------------------
! Goal:
!   Branch between the classical single-shot path (old behaviour) and the new
!   MC ensemble path on the value of qp%enabled (read from namelist).
!   Classical path is UNCHANGED and fully reproducible whenever enabled=.false.
!
! Changes:
!   (1) Add `use mod_quantum_light` and `use mod_ensemble`.
!   (2) Extend read_input() to also parse &quantum_light and return qp.
!   (3) After crystal_info, branch on qp%enabled.
!===============================================================================

program main
  use mod_crystal
  use mod_laser
  use mod_sbes
  use mod_quantum_light, only: qlight_params_t
  use mod_ensemble,      only: run_mc_ensemble
  implicit none

  type(qlight_params_t) :: qp
  character(len=*), parameter :: bsv_out_prefix = 'HHG/bsv_spectrum'

  ! Parse wannier_sbe_input.txt; qp is filled from the new &quantum_light group
  ! (defaults in qlight_params_t apply if the group is absent -- backward-compat).
  call read_input('wannier_sbe_input.txt', ..., qp)

  ! k-mesh + Bloch bands: identical to before, computed once regardless of path.
  call crystal_info(...)

  if (.not. qp%enabled) then
    !---------------------------------------------------------------------------
    ! Classical path -- bit-identical to pre-BSV behaviour.
    !---------------------------------------------------------------------------
    call calc_laser2d()
    call calc_sbes(...)
  else
    !---------------------------------------------------------------------------
    ! Quantum path -- BSV random-phase Monte-Carlo ensemble.
    !---------------------------------------------------------------------------
    ! nw, nt are the frequency and time grid sizes already fixed by your
    ! existing laser + FFT configuration (e.g. n_w_hhg and n_t from mod_sbes).
    call run_mc_ensemble(qp, nw = n_w_hhg, nt = n_t, prefix = bsv_out_prefix)
  end if

end program main

!-------------------------------------------------------------------------------
! Example extension of read_input() -- add inside your existing reader:
!-------------------------------------------------------------------------------
!
!   subroutine read_input(fname, ..., qp)
!     use mod_quantum_light, only: qlight_params_t
!     type(qlight_params_t), intent(out) :: qp
!     logical  :: enabled
!     real(dp) :: I_bar
!     integer  :: n_samples, seed
!     namelist /quantum_light/ enabled, I_bar, n_samples, seed
!
!     ! defaults
!     enabled   = .false.
!     I_bar     = 0.0_dp
!     n_samples = 500
!     seed      = 42
!
!     open(newunit=u, file=fname, status='old', action='read')
!     ! ... read the pre-existing &crystal, &laser, &wannier90, &sbe groups ...
!     rewind(u)
!     read(u, nml=quantum_light, iostat=ios)
!     ! ios /= 0 just means the group is missing -> keep defaults (classical)
!     close(u)
!
!     qp%enabled   = enabled
!     qp%I_bar     = I_bar
!     qp%n_samples = n_samples
!     qp%seed      = seed
!   end subroutine read_input
