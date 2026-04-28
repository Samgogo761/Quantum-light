!===============================================================================
! mod_ensemble
!-------------------------------------------------------------------------------
! Monte-Carlo driver for BSV quantum-light HHG.
! Wraps the SBE solver: for each trajectory, sample (I, phi) from the
! random-phase BSV distribution, run one time evolution, FFT, accumulate
! |J(omega)|^2 with Welford online variance.
!
! Integrated with the new hhg-sbe-solver modules (mod_laser, mod_sbe).
! The new solver uses atomic units internally; the intensity-to-field
! conversion (W/cm^2 → a.u.) is done here.
!===============================================================================
module mod_ensemble
  use mod_quantum_light, only: dp, qlight_params_t, qlight_init, qlight_sample_bsv
  implicit none
  private

  real(dp), parameter :: TWOPI_loc = 2.0_dp * 3.14159265358979323846_dp
  real(dp), parameter :: Wcm2_to_au_loc = 1.0_dp / 3.50944758e16_dp
  real(dp), parameter :: c_au_loc = 137.035999084_dp
  integer,  parameter :: CKPT_EVERY = 50

  type, public :: spec_accum_t
    integer :: n  = 0
    integer :: nw = 0
    complex(dp), allocatable :: sum_Jw(:)
    real(dp),    allocatable :: sum_Jw2(:)
    real(dp),    allocatable :: mean(:)
    real(dp),    allocatable :: M2(:)
  end type spec_accum_t

  public :: run_mc_ensemble
  public :: spec_accum_init, spec_accum_add, spec_accum_dump, spec_accum_finalize

contains

  subroutine spec_accum_init(accum, nw)
    type(spec_accum_t), intent(out) :: accum
    integer,            intent(in)  :: nw
    accum%n = 0; accum%nw = nw
    allocate(accum%sum_Jw(nw));  accum%sum_Jw  = (0.0_dp, 0.0_dp)
    allocate(accum%sum_Jw2(nw)); accum%sum_Jw2 = 0.0_dp
    allocate(accum%mean(nw));    accum%mean    = 0.0_dp
    allocate(accum%M2(nw));      accum%M2      = 0.0_dp
  end subroutine spec_accum_init

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
      delta           = p - accum%mean(iw)
      accum%mean(iw)  = accum%mean(iw) + delta / real(accum%n, dp)
      delta2          = p - accum%mean(iw)
      accum%M2(iw)    = accum%M2(iw)   + delta * delta2
    end do
  end subroutine spec_accum_add

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
    n = accum%n; inv_n = 1.0_dp/real(n,dp); inv_n2 = inv_n*inv_n
    do iw = 1, accum%nw
      s_total = accum%sum_Jw2(iw) * inv_n
      s_coh   = real(accum%sum_Jw(iw)*conjg(accum%sum_Jw(iw)), dp) * inv_n2
      s_quant = s_total - s_coh
      if (n > 1) then
        sigma = sqrt((accum%M2(iw) / real(n-1, dp)) * inv_n)
      else
        sigma = 0.0_dp
      end if
      write(u,'(I8,4ES16.8)') iw, s_total, s_coh, s_quant, sigma
    end do
    close(u)
  end subroutine spec_accum_dump

  subroutine spec_accum_finalize(accum, prefix)
    type(spec_accum_t), intent(in) :: accum
    character(len=*),   intent(in) :: prefix
    call spec_accum_dump(accum, accum%n, trim(prefix) // '_final')
  end subroutine spec_accum_finalize

  subroutine run_mc_ensemble(qp, nt_in, prefix)
    use mod_laser, only: generate_field_sample
    use mod_sbe,   only: run_single_trajectory, init_density_matrix

    type(qlight_params_t), intent(in) :: qp
    integer,               intent(in) :: nt_in
    character(len=*),      intent(in) :: prefix

    type(spec_accum_t) :: accum
    integer  :: i, n_omega
    real(dp) :: I_i, phi_i, E_peak_au
    real(dp), allocatable :: Jt_i(:,:)

    call qlight_init(qp)

    n_omega = nt_in / 2 + 1
    call spec_accum_init(accum, n_omega)
    allocate(Jt_i(nt_in, 2))

    do i = 1, qp%n_samples
      call qlight_sample_bsv(qp, I_i, phi_i)

      E_peak_au = sqrt(2.0_dp * I_i * Wcm2_to_au_loc / c_au_loc)

      call generate_field_sample(E_peak_au, phi_i)
      call run_single_trajectory(Jt_i)

      ! TODO: FFT Jt_i → Jw_i, then spec_accum_add(accum, Jw_i)
      ! For now the BSV path in main.f90 handles FFT directly.

      if (mod(i, CKPT_EVERY) == 0) then
        write(*,'(A,I0,A,I0)') '  MC sample ', i, ' / ', qp%n_samples
      end if
    end do

    deallocate(Jt_i)
    call spec_accum_finalize(accum, prefix)
  end subroutine run_mc_ensemble

end module mod_ensemble
