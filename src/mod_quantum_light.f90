!===============================================================================
! mod_quantum_light
!-------------------------------------------------------------------------------
! Bright Squeezed Vacuum (BSV) random-phase sampler for quantum-light HHG.
!
! Based on Nature Physics 2023 Supplementary (Eqs. 1.32 / II.7): in the
! random-phase approximation, the Husimi-Q distribution of a BSV field
! collapses on the intensity (ponderomotive-energy) axis to an exponential:
!
!     Q_BSV(I) = 1 / (2 I_bar) * exp( -I / (2 I_bar) )
!
! which is trivially sampled by inverse-CDF transform:
!
!     I = -2 * I_bar * ln(1 - u),   u ~ U(0,1)
!     phi = 2*pi * v,               v ~ U(0,1)
!
! Phase is uniform on [0, 2*pi).
!
! This module is self-contained (no external dependencies) so it can be
! compiled and exercised by the stand-alone unit test in tests/.
!===============================================================================
module mod_quantum_light
  implicit none
  private

  integer,  parameter, public :: dp = kind(1.0d0)
  real(dp), parameter, public :: PI    = 3.141592653589793238462643383279502884_dp
  real(dp), parameter, public :: TWOPI = 2.0_dp * PI

  !-----------------------------------------------------------------------------
  ! qlight_params_t
  !   Runtime parameters read from the &quantum_light namelist.
  !   I_bar is BSV mean intensity in W/cm^2 (user-facing unit).
  !-----------------------------------------------------------------------------
  type, public :: qlight_params_t
    logical  :: enabled   = .false.
    real(dp) :: I_bar     = 0.0_dp
    integer  :: n_samples = 500
    integer  :: seed      = 42
  end type qlight_params_t

  public :: qlight_init
  public :: qlight_sample_bsv

contains

  !-----------------------------------------------------------------------------
  ! qlight_init
  !   Seed the Fortran intrinsic PRNG deterministically from p%seed, so
  !   trajectories are reproducible across runs.
  !-----------------------------------------------------------------------------
  subroutine qlight_init(p)
    type(qlight_params_t), intent(in) :: p
    integer :: n, i
    integer, allocatable :: seed_arr(:)

    call random_seed(size = n)
    allocate(seed_arr(n))
    ! Spread the user seed across the full state vector so repeated int
    ! values don't collapse the PRNG into a degenerate sub-cycle.
    do i = 1, n
      seed_arr(i) = p%seed + 37 * (i - 1) + 101 * i * i
    end do
    call random_seed(put = seed_arr)
    deallocate(seed_arr)
  end subroutine qlight_init

  !-----------------------------------------------------------------------------
  ! qlight_sample_bsv
  !   Draw one (I, phi) sample from the random-phase BSV distribution.
  !
  !   I_sample    [W/cm^2]     -- same unit as p%I_bar
  !   phi_sample  [rad]        -- uniform on [0, 2*pi)
  !
  !   Numerical note: u is clamped to [eps, 1 - eps] so that log(1 - u) is
  !   always finite. The upper clamp protects the exponential tail against
  !   overflow; the lower clamp against -0.0 ambiguity.
  !-----------------------------------------------------------------------------
  subroutine qlight_sample_bsv(p, I_sample, phi_sample)
    type(qlight_params_t), intent(in)  :: p
    real(dp),              intent(out) :: I_sample, phi_sample

    real(dp), parameter :: EPS  = epsilon(1.0_dp)
    real(dp), parameter :: UMAX = 1.0_dp - EPS
    real(dp) :: u, v

    call random_number(u)
    if (u < EPS ) u = EPS
    if (u > UMAX) u = UMAX
    I_sample = -2.0_dp * p%I_bar * log(1.0_dp - u)

    call random_number(v)
    phi_sample = TWOPI * v
  end subroutine qlight_sample_bsv

end module mod_quantum_light
