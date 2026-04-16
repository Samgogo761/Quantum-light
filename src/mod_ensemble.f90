!===============================================================================
! mod_ensemble
!-------------------------------------------------------------------------------
! Monte-Carlo driver that wraps the existing (classical) SBE solver into a
! BSV-ensemble HHG pipeline.
!
! For each trajectory i = 1 .. n_samples:
!
!   1. Sample (I_i, phi_i) from the random-phase BSV distribution.
!   2. Convert I_i [W/cm^2] to SI peak E-field via  E = sqrt(2 I / (c eps0)).
!   3. Rebuild the laser field at (E_peak_i, phi_i) via calc_laser2d_sample.
!   4. Re-initialize rho -> |v><v|.
!   5. Run a single RK4 trajectory, returning J_i(t) and J_i(omega).
!   6. Accumulate:
!        sum_Jw        = sum_i J_i(omega)              (coherent sum)
!        sum_Jw2       = sum_i |J_i(omega)|^2          (incoherent sum)
!        Welford M2    -> per-frequency Monte-Carlo variance
!
! Four spectra are finalized:
!   S_total = <|J(omega)|^2>                          (physical HHG output)
!   S_coh   = |<J(omega)>|^2                          (coherent part)
!   S_quant = S_total - S_coh                         (quantum-noise part)
!   sigma(omega)                                      (MC 1-sigma error bar)
!
! INTEGRATION CONTRACT (these symbols must exist in your project):
!   mod_laser   :: calc_laser2d_sample(E_peak_SI, phi_0)    ! rewrites E(t),A(t)
!   mod_sbes    :: init_density_matrix()                    ! rho <- |v><v|
!   mod_sbes    :: sbe_run_single_trajectory(Jt, Jw)        ! RK4 + FFT, no IO
!
! The reference implementations of those three are provided in patches/.
!===============================================================================
module mod_ensemble
  use mod_quantum_light, only: dp, qlight_params_t, qlight_init, qlight_sample_bsv
  implicit none
  private

  !-- Physical constants (SI) ---------------------------------------------------
  real(dp), parameter :: C_SI        = 2.99792458e8_dp        ! m / s
  real(dp), parameter :: EPS0_SI     = 8.8541878128e-12_dp    ! F / m
  real(dp), parameter :: WCM2_TO_WM2 = 1.0e4_dp               ! W/cm^2 -> W/m^2

  !-- Checkpointing cadence -----------------------------------------------------
  integer, parameter :: CKPT_EVERY = 50

  !-----------------------------------------------------------------------------
  ! spec_accum_t
  !   Running accumulators over frequency grid of length nw.
  !-----------------------------------------------------------------------------
  type, public :: spec_accum_t
    integer :: n  = 0
    integer :: nw = 0
    complex(dp), allocatable :: sum_Jw(:)     ! sum_i J_i(w)
    real(dp),    allocatable :: sum_Jw2(:)    ! sum_i |J_i(w)|^2
    real(dp),    allocatable :: mean(:)       ! Welford running mean of |J|^2
    real(dp),    allocatable :: M2(:)         ! Welford running sum-of-squares
  end type spec_accum_t

  public :: run_mc_ensemble
  public :: spec_accum_init, spec_accum_add, spec_accum_dump, spec_accum_finalize

contains

  !-----------------------------------------------------------------------------
  subroutine spec_accum_init(accum, nw)
    type(spec_accum_t), intent(out) :: accum
    integer,            intent(in)  :: nw
    accum%n  = 0
    accum%nw = nw
    allocate(accum%sum_Jw(nw));   accum%sum_Jw  = (0.0_dp, 0.0_dp)
    allocate(accum%sum_Jw2(nw));  accum%sum_Jw2 = 0.0_dp
    allocate(accum%mean(nw));     accum%mean    = 0.0_dp
    allocate(accum%M2(nw));       accum%M2      = 0.0_dp
  end subroutine spec_accum_init

  !-----------------------------------------------------------------------------
  subroutine spec_accum_add(accum, Jw)
    type(spec_accum_t), intent(inout) :: accum
    complex(dp),        intent(in)    :: Jw(:)
    integer  :: iw
    real(dp) :: p, delta, delta2

    accum%n = accum%n + 1
    do iw = 1, accum%nw
      p = real(Jw(iw)*conjg(Jw(iw)), dp)
      accum%sum_Jw(iw)  = accum%sum_Jw(iw)  + Jw(iw)
      accum%sum_Jw2(iw) = accum%sum_Jw2(iw) + p
      ! Welford online mean/variance on |J|^2
      delta           = p - accum%mean(iw)
      accum%mean(iw)  = accum%mean(iw) + delta / real(accum%n, dp)
      delta2          = p - accum%mean(iw)
      accum%M2(iw)    = accum%M2(iw)   + delta * delta2
    end do
  end subroutine spec_accum_add

  !-----------------------------------------------------------------------------
  subroutine spec_accum_dump(accum, i_sample, prefix)
    type(spec_accum_t), intent(in) :: accum
    integer,            intent(in) :: i_sample
    character(len=*),   intent(in) :: prefix
    character(len=256) :: fname
    integer  :: iw, u, n
    real(dp) :: s_total, s_coh, s_quant, sigma, inv_n, inv_n2

    write(fname, '(A,"_ckpt_",I6.6,".dat")') trim(prefix), i_sample
    open(newunit=u, file=trim(fname), status='replace', action='write')
    write(u,'(A,I0)') '# BSV ensemble checkpoint, N = ', accum%n
    write(u,'(A)')    '# iw    S_total         S_coh           S_quant         sigma'

    n      = accum%n
    inv_n  = 1.0_dp / real(n, dp)
    inv_n2 = inv_n * inv_n

    do iw = 1, accum%nw
      s_total = accum%sum_Jw2(iw) * inv_n
      s_coh   = real(accum%sum_Jw(iw) * conjg(accum%sum_Jw(iw)), dp) * inv_n2
      s_quant = s_total - s_coh
      if (n > 1) then
        sigma = sqrt( (accum%M2(iw) / real(n - 1, dp)) * inv_n )
      else
        sigma = 0.0_dp
      end if
      write(u,'(I8,4ES16.8)') iw, s_total, s_coh, s_quant, sigma
    end do
    close(u)
  end subroutine spec_accum_dump

  !-----------------------------------------------------------------------------
  subroutine spec_accum_finalize(accum, prefix)
    type(spec_accum_t), intent(in) :: accum
    character(len=*),   intent(in) :: prefix
    call spec_accum_dump(accum, accum%n, trim(prefix) // '_final')
  end subroutine spec_accum_finalize

  !-----------------------------------------------------------------------------
  ! run_mc_ensemble
  !   Main driver. `nw` must match the FFT frequency grid size that
  !   sbe_run_single_trajectory returns. `prefix` is the output path stem for
  !   checkpoint files (e.g. "HHG/bsv_spectrum").
  !-----------------------------------------------------------------------------
  subroutine run_mc_ensemble(qp, nw, nt, prefix)
    use mod_laser, only: calc_laser2d_sample
    use mod_sbes,  only: sbe_run_single_trajectory, init_density_matrix

    type(qlight_params_t), intent(in) :: qp
    integer,               intent(in) :: nw, nt
    character(len=*),      intent(in) :: prefix

    type(spec_accum_t)       :: accum
    integer                  :: i
    real(dp)                 :: I_i, phi_i, E_peak_i
    real(dp),    allocatable :: Jt_i(:)
    complex(dp), allocatable :: Jw_i(:)

    call qlight_init(qp)
    call spec_accum_init(accum, nw)
    allocate(Jt_i(nt))
    allocate(Jw_i(nw))

    do i = 1, qp%n_samples
      call qlight_sample_bsv(qp, I_i, phi_i)

      ! W/cm^2 -> W/m^2 -> peak E-field (V/m, SI)
      E_peak_i = sqrt( 2.0_dp * I_i * WCM2_TO_WM2 / (C_SI * EPS0_SI) )

      call calc_laser2d_sample(E_peak_i, phi_i)
      call init_density_matrix()
      call sbe_run_single_trajectory(Jt_i, Jw_i)

      call spec_accum_add(accum, Jw_i)

      if (mod(i, CKPT_EVERY) == 0) then
        call spec_accum_dump(accum, i, prefix)
      end if
    end do

    deallocate(Jt_i, Jw_i)
    call spec_accum_finalize(accum, prefix)
  end subroutine run_mc_ensemble

end module mod_ensemble
