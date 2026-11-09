!===============================================================================
! mod_quantum_light
!-------------------------------------------------------------------------------
! Phase-space samplers for quantum-light-driven HHG (layer A / A0).
!
! IMPORTANT NAMING (theory note 2026-07-29 §2.5, §11.2):
!   The historical "BSV" sampler is NOT a fixed-angle single-mode bright
!   squeezed vacuum. It draws a circularly symmetric complex-Gaussian /
!   exponential intensity distribution with a uniform random phase:
!
!       p(I) = 1/(2 I_bar) * exp(-I/(2 I_bar)),   <I> = 2 I_bar
!       phi  ~ U[0, 2*pi)
!
!   That path is retained as the historical benchmark under the name
!       state_type = 'random_phase_exponential'
!   (aliases: 'circular_gaussian_intensity_benchmark', legacy 'bsv').
!
!   True Gaussian Husimi-Q sampling (coherent / thermal / SV / DSV) is
!   provided separately. Q samples generate ANTINORMAL moments:
!       E_Q[|alpha|^2] = <n> + 1
!   Intensity is then rescaled so that the ensemble mean peak intensity
!   equals the user scale I_mean (= 2*I_bar for the exponential path).
!
! This module is self-contained so tests/ can compile it alone.
!===============================================================================
module mod_quantum_light
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private

  integer,  parameter, public :: dp = kind(1.0d0)
  real(dp), parameter, public :: PI    = 3.141592653589793238462643383279502884_dp
  real(dp), parameter, public :: TWOPI = 2.0_dp * PI
  ! Production from_file rejects weight sums outside this absolute window.
  real(dp), parameter :: WEIGHT_SUM_ABS_TOL = 5.0e-8_dp

  !-----------------------------------------------------------------------------
  ! qlight_params_t
  !-----------------------------------------------------------------------------
  type, public :: qlight_params_t
    logical :: enabled = .false.
    ! Intensity scale [W/cm^2]. For random_phase_exponential, sampled mean
    ! intensity is 2*I_bar (historical convention). For Gaussian Q states,
    ! I_mean_drive = 2*I_bar is used as the target ensemble mean intensity.
    real(dp) :: I_bar = 0.0_dp
    integer  :: n_samples = 500
    integer  :: seed = 42
    ! state_type:
    !   random_phase_exponential | circular_gaussian_intensity_benchmark | bsv
    !   coherent | thermal | squeezed_vacuum | displaced_squeezed
    character(64) :: state_type = 'random_phase_exponential'
    ! Single-mode Gaussian parameters (atomic / dimensionless mode units)
    real(dp) :: squeeze_r = 0.0_dp          ! r >= 0
    real(dp) :: squeeze_theta = 0.0_dp      ! compression phase theta_s [rad]
    real(dp) :: alpha0_abs = 0.0_dp         ! |alpha0| for coherent / DSV
    real(dp) :: alpha0_phase = 0.0_dp       ! arg(alpha0) [rad]
    real(dp) :: thermal_nbar = 0.0_dp       ! <n> for thermal state
  end type qlight_params_t

  ! Shared ±N / quadrature node (layer A0.5)
  type, public :: qlight_node_t
    integer  :: id = 0
    real(dp) :: weight = 0.0_dp
    real(dp) :: re_alpha = 0.0_dp
    real(dp) :: im_alpha = 0.0_dp
    real(dp) :: I_drive = 0.0_dp
    real(dp) :: phi_drive = 0.0_dp
  end type qlight_node_t

  public :: qlight_init
  public :: qlight_normalize_state_type
  public :: qlight_sample_drive
  public :: qlight_sample_random_phase_exponential
  public :: qlight_sample_bsv
  public :: qlight_build_VQ_single_mode
  public :: qlight_sample_Q_alpha
  public :: qlight_alpha_to_drive
  public :: qlight_physical_nbar
  public :: qlight_Q_abs2_expectation
  public :: qlight_validate_Q0_report
  public :: qlight_normalize_sampling_mode
  public :: qlight_build_nodes
  public :: qlight_write_nodes_manifest
  public :: qlight_read_nodes_manifest
  public :: qlight_validate_nodes_moments
  public :: qlight_validate_nodes_integrity
  public :: qlight_parse_propagate_ids
  public :: qlight_select_nodes_by_ids

contains

  !-----------------------------------------------------------------------------
  subroutine qlight_init(p)
    type(qlight_params_t), intent(in) :: p
    integer :: n, i
    integer, allocatable :: seed_arr(:)

    call random_seed(size = n)
    allocate(seed_arr(n))
    do i = 1, n
      seed_arr(i) = p%seed + 37 * (i - 1) + 101 * i * i
    end do
    call random_seed(put = seed_arr)
    deallocate(seed_arr)
  end subroutine qlight_init

  !-----------------------------------------------------------------------------
  function qlight_normalize_state_type(raw) result(name)
    character(*), intent(in) :: raw
    character(64) :: name
    character(64) :: s
    integer :: i, n

    s = adjustl(raw)
    n = len_trim(s)
    do i = 1, n
      if (s(i:i) >= 'A' .and. s(i:i) <= 'Z') then
        s(i:i) = achar(iachar(s(i:i)) + 32)
      end if
    end do

    select case (trim(s))
    case ('random_phase_exponential', 'circular_gaussian_intensity_benchmark', &
          'bsv', 'rpe', 'thermal_like_complex_gaussian')
      name = 'random_phase_exponential'
    case ('coherent', 'coh')
      name = 'coherent'
    case ('thermal', 'th')
      name = 'thermal'
    case ('squeezed_vacuum', 'sv', 'smsv', 'single_mode_squeezed_vacuum')
      name = 'squeezed_vacuum'
    case ('displaced_squeezed', 'dsv', 'displaced_squeezed_vacuum')
      name = 'displaced_squeezed'
    case default
      write(*,*) 'ERROR: unknown quantum-light state_type: ', trim(raw)
      write(*,*) '  allowed: random_phase_exponential, coherent, thermal,'
      write(*,*) '           squeezed_vacuum, displaced_squeezed'
      error stop 1
    end select
  end function qlight_normalize_state_type

  !-----------------------------------------------------------------------------
  ! Unified drive sampler used by main / ensemble.
  ! Returns peak intensity [W/cm^2] and CEP [rad], plus optional alpha.
  !-----------------------------------------------------------------------------
  subroutine qlight_sample_drive(p, I_sample, phi_sample, alpha_out)
    type(qlight_params_t), intent(in)  :: p
    real(dp),              intent(out) :: I_sample, phi_sample
    complex(dp), optional, intent(out) :: alpha_out

    character(64) :: st
    complex(dp) :: alpha
    real(dp) :: I_mean

    st = qlight_normalize_state_type(p%state_type)
    I_mean = 2.0_dp * p%I_bar

    select case (trim(st))
    case ('random_phase_exponential')
      call qlight_sample_random_phase_exponential(p, I_sample, phi_sample)
      alpha = sqrt(max(I_sample, 0.0_dp) / max(I_mean, tiny(1.0_dp))) * &
              exp(cmplx(0.0_dp, phi_sample, dp))
    case default
      call qlight_sample_Q_alpha(p, alpha)
      call qlight_alpha_to_drive(p, alpha, I_sample, phi_sample)
    end select

    if (present(alpha_out)) alpha_out = alpha
  end subroutine qlight_sample_drive

  !-----------------------------------------------------------------------------
  ! Historical circular / exponential intensity sampler (NOT fixed-angle SV).
  !-----------------------------------------------------------------------------
  subroutine qlight_sample_random_phase_exponential(p, I_sample, phi_sample)
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
  end subroutine qlight_sample_random_phase_exponential

  ! Legacy alias kept for old call sites / docs; prefer the explicit name.
  subroutine qlight_sample_bsv(p, I_sample, phi_sample)
    type(qlight_params_t), intent(in)  :: p
    real(dp),              intent(out) :: I_sample, phi_sample
    call qlight_sample_random_phase_exponential(p, I_sample, phi_sample)
  end subroutine qlight_sample_bsv

  !-----------------------------------------------------------------------------
  ! Physical <n> and E_Q[|alpha|^2] = <n>+1 for single-mode Gaussian states.
  !-----------------------------------------------------------------------------
  pure real(dp) function qlight_physical_nbar(p) result(nbar)
    type(qlight_params_t), intent(in) :: p
    character(64) :: st
    real(dp) :: sh, ch

    ! pure functions cannot call non-pure normalize; inline a safe subset
    st = adjustl(p%state_type)
    select case (trim(st))
    case ('coherent', 'coh')
      nbar = p%alpha0_abs**2
    case ('thermal', 'th')
      nbar = max(p%thermal_nbar, 0.0_dp)
    case ('squeezed_vacuum', 'sv', 'smsv', 'single_mode_squeezed_vacuum')
      sh = sinh(p%squeeze_r)
      nbar = sh * sh
    case ('displaced_squeezed', 'dsv', 'displaced_squeezed_vacuum')
      sh = sinh(p%squeeze_r)
      ch = cosh(p%squeeze_r)
      ! <n> = |alpha0|^2 + sinh^2 r   (standard D(alpha)S(xi)|0>)
      nbar = p%alpha0_abs**2 + sh * sh
    case default
      nbar = -1.0_dp
    end select
  end function qlight_physical_nbar

  pure real(dp) function qlight_Q_abs2_expectation(p) result(eabs2)
    type(qlight_params_t), intent(in) :: p
    real(dp) :: nbar
    nbar = qlight_physical_nbar(p)
    if (nbar < 0.0_dp) then
      eabs2 = 1.0_dp
    else
      eabs2 = nbar + 1.0_dp
    end if
  end function qlight_Q_abs2_expectation

  !-----------------------------------------------------------------------------
  ! Build single-mode Husimi covariance V_Q (2x2) in (x,p) quadrature basis.
  ! Convention: vacuum Var(x)=Var(p)=1/2, V_Q = V + (1/2) I.
  ! SV ellipse is rotated by theta_s/2 (NOT theta_s).
  !-----------------------------------------------------------------------------
  subroutine qlight_build_VQ_single_mode(p, VQ, d_vec)
    type(qlight_params_t), intent(in)  :: p
    real(dp),              intent(out) :: VQ(2,2)
    real(dp),              intent(out) :: d_vec(2)

    character(64) :: st
    real(dp) :: vx, vp, c, s, rot(2,2), Vdiag(2,2), nbar
    complex(dp) :: alpha0

    st = qlight_normalize_state_type(p%state_type)
    VQ = 0.0_dp
    d_vec = 0.0_dp
    alpha0 = p%alpha0_abs * exp(cmplx(0.0_dp, p%alpha0_phase, dp))

    select case (trim(st))
    case ('coherent')
      ! Physical V = (1/2) I  =>  V_Q = I
      VQ(1,1) = 1.0_dp
      VQ(2,2) = 1.0_dp
      d_vec(1) = sqrt(2.0_dp) * real(alpha0, dp)
      d_vec(2) = sqrt(2.0_dp) * aimag(alpha0)

    case ('thermal')
      ! Physical V = (<n>+1/2) I  =>  V_Q = (<n>+1) I
      nbar = max(p%thermal_nbar, 0.0_dp)
      VQ(1,1) = nbar + 1.0_dp
      VQ(2,2) = nbar + 1.0_dp

    case ('squeezed_vacuum', 'displaced_squeezed')
      ! Eq. (2.23): V_Q = (1/2) diag(1+e^{-2r}, 1+e^{2r}), then rotate by theta_s/2
      vx = 0.5_dp * (1.0_dp + exp(-2.0_dp * p%squeeze_r))
      vp = 0.5_dp * (1.0_dp + exp(+2.0_dp * p%squeeze_r))
      Vdiag = 0.0_dp
      Vdiag(1,1) = vx
      Vdiag(2,2) = vp
      c = cos(0.5_dp * p%squeeze_theta)
      s = sin(0.5_dp * p%squeeze_theta)
      rot(1,1) = c;  rot(1,2) = -s
      rot(2,1) = s;  rot(2,2) =  c
      VQ = matmul(rot, matmul(Vdiag, transpose(rot)))
      if (trim(st) == 'displaced_squeezed') then
        d_vec(1) = sqrt(2.0_dp) * real(alpha0, dp)
        d_vec(2) = sqrt(2.0_dp) * aimag(alpha0)
      end if

    case default
      write(*,*) 'ERROR: qlight_build_VQ_single_mode not defined for state_type=', trim(st)
      error stop 1
    end select
  end subroutine qlight_build_VQ_single_mode

  !-----------------------------------------------------------------------------
  subroutine qlight_sample_Q_alpha(p, alpha)
    type(qlight_params_t), intent(in)  :: p
    complex(dp),           intent(out) :: alpha

    real(dp) :: VQ(2,2), d_vec(2), L(2,2), z(2), xi(2)

    call qlight_build_VQ_single_mode(p, VQ, d_vec)
    call chol2_sym(VQ, L)
    call box_muller(z(1), z(2))
    xi = d_vec + matmul(L, z)
    alpha = cmplx(xi(1), xi(2), dp) / sqrt(2.0_dp)
  end subroutine qlight_sample_Q_alpha

  !-----------------------------------------------------------------------------
  ! Map a Q-sample alpha onto classical drive (I, phi).
  ! I = I_mean * |alpha|^2 / E_Q[|alpha|^2], so <I> -> I_mean = 2*I_bar.
  !-----------------------------------------------------------------------------
  subroutine qlight_alpha_to_drive(p, alpha, I_sample, phi_sample)
    type(qlight_params_t), intent(in)  :: p
    complex(dp),           intent(in)  :: alpha
    real(dp),              intent(out) :: I_sample, phi_sample

    real(dp) :: I_mean, eabs2, abs2

    I_mean = 2.0_dp * p%I_bar
    eabs2  = qlight_Q_abs2_expectation(p)
    abs2   = real(alpha * conjg(alpha), dp)
    I_sample = I_mean * abs2 / max(eabs2, tiny(1.0_dp))
    phi_sample = atan2(aimag(alpha), real(alpha, dp))
    if (phi_sample < 0.0_dp) phi_sample = phi_sample + TWOPI
  end subroutine qlight_alpha_to_drive

  !-----------------------------------------------------------------------------
  ! Q0 moment check report for Gaussian states (Monte Carlo).
  ! Writes a small text report; returns pass=.true. if tolerances met.
  !-----------------------------------------------------------------------------
  subroutine qlight_validate_Q0_report(p, n_draw, report_file, pass)
    type(qlight_params_t), intent(in)  :: p
    integer,               intent(in)  :: n_draw
    character(*),          intent(in)  :: report_file
    logical,               intent(out) :: pass

    integer :: i, u
    complex(dp) :: alpha
    real(dp) :: VQ(2,2), d_vec(2), xi(2)
    real(dp) :: mean_xi(2), cov(2,2), dx(2)
    real(dp) :: mean_N, mean_M_re, mean_M_im, abs2
    real(dp) :: nbar, M_abs, M_phase
    real(dp) :: err_d, err_VQ, err_N, err_M
    real(dp), parameter :: TOL_D = 5.0e-2_dp
    real(dp), parameter :: TOL_V = 8.0e-2_dp
    real(dp), parameter :: TOL_N = 8.0e-2_dp
    real(dp), parameter :: TOL_M = 8.0e-2_dp
    character(64) :: st

    st = qlight_normalize_state_type(p%state_type)
    if (trim(st) == 'random_phase_exponential') then
      open(newunit=u, file=report_file, status='replace', action='write')
      write(u,'(A)') '# Q0 check skipped: random_phase_exponential is not a Gaussian Q state'
      write(u,'(A)') 'pass = true (N/A)'
      close(u)
      pass = .true.
      return
    end if

    call qlight_build_VQ_single_mode(p, VQ, d_vec)

    mean_xi = 0.0_dp
    cov = 0.0_dp
    mean_N = 0.0_dp
    mean_M_re = 0.0_dp
    mean_M_im = 0.0_dp

    do i = 1, n_draw
      call qlight_sample_Q_alpha(p, alpha)
      xi(1) = sqrt(2.0_dp) * real(alpha, dp)
      xi(2) = sqrt(2.0_dp) * aimag(alpha)
      mean_xi = mean_xi + xi
      abs2 = real(alpha * conjg(alpha), dp)
      ! Antinormal: E_Q[|alpha|^2] = N+1  =>  proxy N_Q = |alpha|^2 - 1
      mean_N = mean_N + (abs2 - 1.0_dp)
      mean_M_re = mean_M_re + real(alpha * alpha, dp)
      mean_M_im = mean_M_im + aimag(alpha * alpha)
    end do

    mean_xi = mean_xi / real(n_draw, dp)
    mean_N = mean_N / real(n_draw, dp)
    mean_M_re = mean_M_re / real(n_draw, dp)
    mean_M_im = mean_M_im / real(n_draw, dp)

    ! Second pass for covariance (simple; n_draw is moderate in tests)
    cov = 0.0_dp
    do i = 1, n_draw
      call qlight_sample_Q_alpha(p, alpha)
      xi(1) = sqrt(2.0_dp) * real(alpha, dp)
      xi(2) = sqrt(2.0_dp) * aimag(alpha)
      dx = xi - mean_xi
      cov(1,1) = cov(1,1) + dx(1)*dx(1)
      cov(1,2) = cov(1,2) + dx(1)*dx(2)
      cov(2,1) = cov(2,1) + dx(2)*dx(1)
      cov(2,2) = cov(2,2) + dx(2)*dx(2)
    end do
    cov = cov / real(n_draw, dp)

    err_d = sqrt(sum((mean_xi - d_vec)**2)) / max(1.0_dp, sqrt(sum(d_vec**2)))
    err_VQ = sqrt(sum((cov - VQ)**2)) / max(1.0_dp, sqrt(sum(VQ**2)))

    nbar = qlight_physical_nbar(p)
    err_N = abs(mean_N - nbar) / max(1.0_dp, abs(nbar))

    ! Expected M = <a a> = -e^{i theta_s} sinh r cosh r  for SV (and DSV fluctuation part)
    M_abs = sinh(p%squeeze_r) * cosh(p%squeeze_r)
    M_phase = p%squeeze_theta
    ! For pure SV / DSV the anomalous moment of the state is M_in; Q reproduces M
    ! (antinormal does not add to <aa>).
    if (trim(st) == 'squeezed_vacuum') then
      err_M = abs(cmplx(mean_M_re, mean_M_im, dp) + &
                  exp(cmplx(0.0_dp, M_phase, dp)) * M_abs) / max(1.0_dp, M_abs)
    else
      err_M = 0.0_dp
    end if

    pass = (err_d <= TOL_D) .and. (err_VQ <= TOL_V) .and. (err_N <= TOL_N)
    if (trim(st) == 'squeezed_vacuum' .and. p%squeeze_r > 1.0e-8_dp) then
      pass = pass .and. (err_M <= TOL_M)
    end if

    open(newunit=u, file=report_file, status='replace', action='write')
    write(u,'(A)') '# Q0 Husimi-Q moment validation (layer A0)'
    write(u,'(A,A)') 'state_type = ', trim(st)
    write(u,'(A,I0)') 'n_draw = ', n_draw
    write(u,'(A,2ES16.8)') 'd_target = ', d_vec
    write(u,'(A,2ES16.8)') 'd_sample = ', mean_xi
    write(u,'(A,ES16.8)') 'err_d = ', err_d
    write(u,'(A)') 'VQ_target ='
    write(u,'(2ES16.8)') VQ(1,1), VQ(1,2)
    write(u,'(2ES16.8)') VQ(2,1), VQ(2,2)
    write(u,'(A)') 'VQ_sample ='
    write(u,'(2ES16.8)') cov(1,1), cov(1,2)
    write(u,'(2ES16.8)') cov(2,1), cov(2,2)
    write(u,'(A,ES16.8)') 'err_VQ = ', err_VQ
    write(u,'(A,ES16.8)') 'N_target = ', nbar
    write(u,'(A,ES16.8)') 'N_sample = ', mean_N
    write(u,'(A,ES16.8)') 'err_N = ', err_N
    write(u,'(A,2ES16.8)') 'M_sample = ', mean_M_re, mean_M_im
    write(u,'(A,ES16.8)') 'err_M = ', err_M
    write(u,'(A,L1)') 'pass = ', pass
    close(u)
  end subroutine qlight_validate_Q0_report

  ! --- helpers ----------------------------------------------------------------
  subroutine box_muller(z1, z2)
    real(dp), intent(out) :: z1, z2
    real(dp), parameter :: EPS = epsilon(1.0_dp)
    real(dp) :: u1, u2, r, theta

    call random_number(u1)
    call random_number(u2)
    if (u1 < EPS) u1 = EPS
    r = sqrt(-2.0_dp * log(u1))
    theta = TWOPI * u2
    z1 = r * cos(theta)
    z2 = r * sin(theta)
  end subroutine box_muller

  subroutine chol2_sym(A, L)
    ! Cholesky of 2x2 SPD matrix: A = L L^T, L lower-triangular.
    real(dp), intent(in)  :: A(2,2)
    real(dp), intent(out) :: L(2,2)
    real(dp) :: a11, a21, a22

    a11 = A(1,1)
    a21 = 0.5_dp * (A(2,1) + A(1,2))
    a22 = A(2,2)
    if (a11 <= 0.0_dp) then
      write(*,*) 'ERROR: chol2_sym failed, A(1,1)<=0'
      error stop 1
    end if
    L = 0.0_dp
    L(1,1) = sqrt(a11)
    L(2,1) = a21 / L(1,1)
    L(2,2) = a22 - L(2,1)*L(2,1)
    if (L(2,2) <= 0.0_dp) then
      write(*,*) 'ERROR: chol2_sym failed, non-SPD VQ'
      error stop 1
    end if
    L(2,2) = sqrt(L(2,2))
  end subroutine chol2_sym

  !===========================================================================
  ! Node manifests / deterministic quadrature (A0.5)
  !===========================================================================
  function qlight_normalize_sampling_mode(raw) result(name)
    character(*), intent(in) :: raw
    character(64) :: name, s
    integer :: i, n
    s = adjustl(raw)
    n = len_trim(s)
    do i = 1, n
      if (s(i:i) >= 'A' .and. s(i:i) <= 'Z') s(i:i) = achar(iachar(s(i:i)) + 32)
    end do
    select case (trim(s))
    case ('mc', 'monte_carlo', 'random')
      name = 'mc'
    case ('gauss_hermite', 'gh', 'quadrature_gh')
      name = 'gauss_hermite'
    case ('exponential_quad', 'exp_quad', 'rpe_quad')
      name = 'exponential_quad'
    case ('from_file', 'file', 'manifest')
      name = 'from_file'
    case default
      write(*,*) 'ERROR: unknown bsv_sampling_mode: ', trim(raw)
      error stop 1
    end select
  end function qlight_normalize_sampling_mode

  subroutine qlight_build_nodes(p, mode_in, gh_order, nodes_file, nodes)
    type(qlight_params_t), intent(in) :: p
    character(*),          intent(in) :: mode_in
    integer,               intent(in) :: gh_order
    character(*),          intent(in) :: nodes_file
    type(qlight_node_t), allocatable, intent(out) :: nodes(:)
    character(64) :: mode

    mode = qlight_normalize_sampling_mode(mode_in)
    select case (trim(mode))
    case ('from_file')
      call qlight_read_nodes_manifest(nodes_file, nodes)
    case ('mc')
      call build_nodes_mc(p, nodes)
    case ('gauss_hermite')
      call build_nodes_gauss_hermite(p, gh_order, nodes)
    case ('exponential_quad')
      call build_nodes_exponential_quad(p, nodes)
    end select
  end subroutine qlight_build_nodes

  subroutine qlight_write_nodes_manifest(filename, nodes, state_label, mode_label)
    character(*), intent(in) :: filename, state_label, mode_label
    type(qlight_node_t), intent(in) :: nodes(:)
    integer :: u, i
    real(dp) :: wsum
    wsum = sum(nodes%weight)
    open(newunit=u, file=filename, status='replace', action='write')
    write(u,'(A)') '# quantum-light node manifest (shared ±N / quadrature)'
    write(u,'(A,A)') '# state_type = ', trim(state_label)
    write(u,'(A,A)') '# sampling_mode = ', trim(mode_label)
    write(u,'(A)') '# NOTE: I_drive uses equal-mean-intensity mapping I = I_mean*|α|^2/E_Q[|α|^2]'
    write(u,'(A)') '#       (not experimental BSV pulse-energy calibration).'
    write(u,'(A,I0)') '# n_nodes = ', size(nodes)
    write(u,'(A,ES16.8)') '# sum_weight = ', wsum
    write(u,'(A)') '# sample_id  weight  Re(alpha)  Im(alpha)  I_drive  phi_drive'
    do i = 1, size(nodes)
      ! ES25.17 keeps alpha <-> (I,phi) remaps stable under from_file round-trip.
      write(u,'(I8,5ES25.17)') nodes(i)%id, nodes(i)%weight, &
        nodes(i)%re_alpha, nodes(i)%im_alpha, nodes(i)%I_drive, nodes(i)%phi_drive
    end do
    close(u)
  end subroutine qlight_write_nodes_manifest

  subroutine qlight_read_nodes_manifest(filename, nodes)
    character(*), intent(in) :: filename
    type(qlight_node_t), allocatable, intent(out) :: nodes(:)
    integer :: u, ios, id, i, j
    real(dp) :: w, ra, ia, Ii, phi
    character(512) :: line
    type(qlight_node_t), allocatable :: tmp(:), grow(:)
    integer :: cap, nread
    logical :: exists

    inquire(file=trim(filename), exist=exists)
    if (.not. exists) then
      write(*,*) 'ERROR: nodes manifest not found: ', trim(filename)
      error stop 1
    end if
    cap = 64
    allocate(tmp(cap))
    nread = 0
    open(newunit=u, file=trim(filename), status='old', action='read')
    do
      read(u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle
      if (line(1:1) == '#') cycle
      read(line, *, iostat=ios) id, w, ra, ia, Ii, phi
      if (ios /= 0) then
        write(*,*) 'ERROR: malformed nodes manifest row: ', trim(line)
        error stop 1
      end if
      ! Reject NaN and +/-Inf (x/=x only catches NaN).
      if (.not. ieee_is_finite(w) .or. .not. ieee_is_finite(ra) .or. &
          .not. ieee_is_finite(ia) .or. .not. ieee_is_finite(Ii) .or. &
          .not. ieee_is_finite(phi)) then
        write(*,*) 'ERROR: non-finite value in nodes manifest row: ', trim(line)
        error stop 1
      end if
      if (w < 0.0_dp) then
        write(*,*) 'ERROR: negative weight in nodes manifest, id=', id
        error stop 1
      end if
      if (Ii < 0.0_dp) then
        write(*,*) 'ERROR: negative I_drive in nodes manifest, id=', id
        error stop 1
      end if
      nread = nread + 1
      if (nread > cap) then
        allocate(grow(cap*2))
        grow(1:cap) = tmp
        deallocate(tmp)
        call move_alloc(grow, tmp)
        cap = size(tmp)
      end if
      tmp(nread)%id = id
      tmp(nread)%weight = w
      tmp(nread)%re_alpha = ra
      tmp(nread)%im_alpha = ia
      tmp(nread)%I_drive = Ii
      tmp(nread)%phi_drive = phi
    end do
    close(u)
    if (nread <= 0) then
      write(*,*) 'ERROR: no data rows in nodes manifest: ', trim(filename)
      error stop 1
    end if
    ! duplicate ids
    do i = 1, nread - 1
      do j = i + 1, nread
        if (tmp(i)%id == tmp(j)%id) then
          write(*,*) 'ERROR: duplicate node id in manifest: ', tmp(i)%id
          error stop 1
        end if
      end do
    end do
    allocate(nodes(nread))
    nodes = tmp(1:nread)
    deallocate(tmp)
    w = sum(nodes%weight)
    if (.not. ieee_is_finite(w) .or. w <= 0.0_dp) then
      write(*,*) 'ERROR: non-positive/non-finite weight sum in manifest'
      error stop 1
    end if
    ! Production from_file path: refuse silent repair of bad weight sums.
    if (abs(w - 1.0_dp) > WEIGHT_SUM_ABS_TOL) then
      write(*,'(A,ES16.8,A,ES16.8)') &
        'ERROR: manifest weight sum =', w, &
        ' outside 1 +/- ', WEIGHT_SUM_ABS_TOL
      write(*,*) '  Refuse to renormalize from_file manifests in production.'
      error stop 1
    end if
  end subroutine qlight_read_nodes_manifest

  subroutine qlight_validate_nodes_integrity(p, nodes, report_file, pass)
    ! Check alpha <-> (I,phi) consistency under equal-mean-intensity mapping,
    ! plus basic finite/nonnegativity (defense in depth after read).
    type(qlight_params_t), intent(in) :: p
    type(qlight_node_t),   intent(in) :: nodes(:)
    character(*),          intent(in) :: report_file
    logical,               intent(out) :: pass
    integer :: i, u, n_bad
    real(dp) :: I_chk, phi_chk, dI, dphi, abs2
    complex(dp) :: alpha
    ! Tolerant enough for legacy ES16.8 manifests; new writes use ES25.17.
    real(dp), parameter :: RTOL_I = 1.0e-6_dp
    real(dp), parameter :: ATOL_I = 1.0e-6_dp
    real(dp), parameter :: ATOL_PHI = 1.0e-6_dp

    n_bad = 0
    pass = .true.
    open(newunit=u, file=report_file, status='replace', action='write')
    write(u,'(A)') '# nodes integrity: finite / nonneg / alpha->(I,phi) map'
    write(u,'(A,I0)') 'n_nodes = ', size(nodes)
    do i = 1, size(nodes)
      if (nodes(i)%weight < 0.0_dp .or. nodes(i)%I_drive < 0.0_dp) then
        n_bad = n_bad + 1
        write(u,'(A,I0)') 'FAIL nonneg id=', nodes(i)%id
        cycle
      end if
      if (.not. ieee_is_finite(nodes(i)%weight) .or. &
          .not. ieee_is_finite(nodes(i)%I_drive) .or. &
          .not. ieee_is_finite(nodes(i)%phi_drive) .or. &
          .not. ieee_is_finite(nodes(i)%re_alpha) .or. &
          .not. ieee_is_finite(nodes(i)%im_alpha)) then
        n_bad = n_bad + 1
        write(u,'(A,I0)') 'FAIL nonfinite id=', nodes(i)%id
        cycle
      end if
      alpha = cmplx(nodes(i)%re_alpha, nodes(i)%im_alpha, dp)
      abs2 = real(alpha * conjg(alpha), dp)
      if (.not. ieee_is_finite(abs2)) then
        n_bad = n_bad + 1
        write(u,'(A,I0)') 'FAIL nonfinite |alpha|^2 id=', nodes(i)%id
        cycle
      end if
      call qlight_alpha_to_drive(p, alpha, I_chk, phi_chk)
      dI = abs(I_chk - nodes(i)%I_drive) / max(abs(nodes(i)%I_drive), ATOL_I)
      dphi = abs(modulo(phi_chk - nodes(i)%phi_drive + PI, TWOPI) - PI)
      ! RPE exponential_quad stores alpha as sqrt(I/I_mean)*e^{i phi}; mapping
      ! through Q expectation may differ. Only enforce for Gaussian Q states.
      if (qlight_normalize_state_type(p%state_type) /= 'random_phase_exponential') then
        if (dI > RTOL_I .or. dphi > ATOL_PHI) then
          n_bad = n_bad + 1
          write(u,'(A,I0,2ES16.8)') 'FAIL map id=', nodes(i)%id, dI, dphi
        end if
      end if
    end do
    pass = (n_bad == 0)
    write(u,'(A,I0)') 'n_bad = ', n_bad
    write(u,'(A,L1)') 'pass = ', pass
    close(u)
  end subroutine qlight_validate_nodes_integrity

  subroutine qlight_validate_nodes_moments(p, nodes, report_file, pass)
    type(qlight_params_t), intent(in) :: p
    type(qlight_node_t),   intent(in) :: nodes(:)
    character(*),          intent(in) :: report_file
    logical,               intent(out) :: pass
    integer :: i, u
    real(dp) :: wsum, VQ(2,2), d_vec(2), mean_xi(2), cov(2,2), xi(2), dx(2), err_d, err_v
    character(64) :: st
    st = qlight_normalize_state_type(p%state_type)
    wsum = sum(nodes%weight)
    pass = abs(wsum - 1.0_dp) < 1.0e-8_dp
    open(newunit=u, file=report_file, status='replace', action='write')
    write(u,'(A)') '# node-moment validation'
    write(u,'(A,A)') 'state_type = ', trim(st)
    write(u,'(A,I0)') 'n_nodes = ', size(nodes)
    write(u,'(A,ES16.8)') 'sum_weight = ', wsum
    if (trim(st) == 'random_phase_exponential') then
      write(u,'(A)') 'pass_moments = N/A (exponential_quad / RPE)'
      write(u,'(A,L1)') 'pass = ', pass
      close(u)
      return
    end if
    call qlight_build_VQ_single_mode(p, VQ, d_vec)
    mean_xi = 0.0_dp
    cov = 0.0_dp
    do i = 1, size(nodes)
      xi(1) = sqrt(2.0_dp) * nodes(i)%re_alpha
      xi(2) = sqrt(2.0_dp) * nodes(i)%im_alpha
      mean_xi = mean_xi + nodes(i)%weight * xi
    end do
    do i = 1, size(nodes)
      xi(1) = sqrt(2.0_dp) * nodes(i)%re_alpha
      xi(2) = sqrt(2.0_dp) * nodes(i)%im_alpha
      dx = xi - mean_xi
      cov(1,1) = cov(1,1) + nodes(i)%weight * dx(1)*dx(1)
      cov(1,2) = cov(1,2) + nodes(i)%weight * dx(1)*dx(2)
      cov(2,1) = cov(2,1) + nodes(i)%weight * dx(2)*dx(1)
      cov(2,2) = cov(2,2) + nodes(i)%weight * dx(2)*dx(2)
    end do
    err_d = sqrt(sum((mean_xi - d_vec)**2)) / max(1.0_dp, sqrt(sum(d_vec**2)))
    err_v = sqrt(sum((cov - VQ)**2)) / max(1.0_dp, sqrt(sum(VQ**2)))
    ! GH low order is approximate; loose gate for smoke
    pass = pass .and. (err_d < 0.25_dp) .and. (err_v < 0.35_dp)
    write(u,'(A,ES16.8)') 'err_d = ', err_d
    write(u,'(A,ES16.8)') 'err_VQ = ', err_v
    write(u,'(A,L1)') 'pass = ', pass
    close(u)
  end subroutine qlight_validate_nodes_moments

  subroutine build_nodes_mc(p, nodes)
    type(qlight_params_t), intent(in) :: p
    type(qlight_node_t), allocatable, intent(out) :: nodes(:)
    integer :: i, n
    real(dp) :: I_i, phi_i
    complex(dp) :: alpha
    n = max(p%n_samples, 1)
    allocate(nodes(n))
    do i = 1, n
      call qlight_sample_drive(p, I_i, phi_i, alpha)
      nodes(i)%id = i
      nodes(i)%weight = 1.0_dp / real(n, dp)
      nodes(i)%re_alpha = real(alpha, dp)
      nodes(i)%im_alpha = aimag(alpha)
      nodes(i)%I_drive = I_i
      nodes(i)%phi_drive = phi_i
    end do
  end subroutine build_nodes_mc

  subroutine build_nodes_gauss_hermite(p, gh_order, nodes)
    type(qlight_params_t), intent(in) :: p
    integer, intent(in) :: gh_order
    type(qlight_node_t), allocatable, intent(out) :: nodes(:)
    real(dp) :: x(7), w(7), VQ(2,2), d_vec(2), L(2,2), z(2), xi(2)
    complex(dp) :: alpha
    real(dp) :: I_i, phi_i, w2, wsum
    integer :: n1, i, j, k, ntot
    character(64) :: st

    st = qlight_normalize_state_type(p%state_type)
    if (trim(st) == 'random_phase_exponential') then
      write(*,*) 'ERROR: gauss_hermite requires a Gaussian Q state_type, not RPE'
      error stop 1
    end if
    call gh_nodes_weights(gh_order, n1, x, w)
    ntot = n1 * n1
    allocate(nodes(ntot))
    call qlight_build_VQ_single_mode(p, VQ, d_vec)
    call chol2_sym(VQ, L)
    k = 0
    wsum = 0.0_dp
    do i = 1, n1
      do j = 1, n1
        k = k + 1
        ! Physicist GH: ∫ e^{-u^2} f(u) du ≈ Σ w f(x)
        ! Standard normal: z = √2 u, weight factor w/√π each dim
        z(1) = sqrt(2.0_dp) * x(i)
        z(2) = sqrt(2.0_dp) * x(j)
        w2 = (w(i) / sqrt(PI)) * (w(j) / sqrt(PI))
        xi = d_vec + matmul(L, z)
        alpha = cmplx(xi(1), xi(2), dp) / sqrt(2.0_dp)
        call qlight_alpha_to_drive(p, alpha, I_i, phi_i)
        nodes(k)%id = k
        nodes(k)%weight = w2
        nodes(k)%re_alpha = real(alpha, dp)
        nodes(k)%im_alpha = aimag(alpha)
        nodes(k)%I_drive = I_i
        nodes(k)%phi_drive = phi_i
        wsum = wsum + w2
      end do
    end do
    nodes%weight = nodes%weight / wsum
  end subroutine build_nodes_gauss_hermite

  subroutine build_nodes_exponential_quad(p, nodes)
    type(qlight_params_t), intent(in) :: p
    type(qlight_node_t), allocatable, intent(out) :: nodes(:)
    ! 16-node intensity quadrature from deploy/bsv_quad_nodes.dat (fixed CEP)
    real(dp), parameter :: I0 = 2.0e11_dp
    real(dp), parameter :: x(16) = (/ &
      0.119130_dp, 0.610001_dp, 1.423403_dp, 2.449696_dp, &
      3.550304_dp, 4.576597_dp, 5.389999_dp, 5.880870_dp, &
      6.238261_dp, 7.220001_dp, 8.846806_dp, 10.899392_dp, &
      13.100608_dp, 15.153194_dp, 16.779999_dp, 17.761739_dp /)
    real(dp), parameter :: ww(16) = (/ &
      2.695793e-01_dp, 3.624926e-01_dp, 2.267091e-01_dp, 9.392039e-02_dp, &
      3.124439e-02_dp, 9.683984e-03_dp, 3.043491e-03_dp, 8.479984e-04_dp, &
      1.186348e-03_dp, 9.764328e-04_dp, 2.707423e-04_dp, 4.019144e-05_dp, &
      4.447927e-06_dp, 4.939992e-07_dp, 6.883163e-08_dp, 1.173894e-08_dp /)
    integer :: i
    real(dp) :: wsum, I_mean, abs_alpha
    character(64) :: st
    st = qlight_normalize_state_type(p%state_type)
    if (trim(st) /= 'random_phase_exponential') then
      write(*,*) 'ERROR: exponential_quad only for random_phase_exponential'
      error stop 1
    end if
    allocate(nodes(16))
    wsum = sum(ww)
    I_mean = 2.0_dp * p%I_bar
    do i = 1, 16
      nodes(i)%id = i
      nodes(i)%weight = ww(i) / wsum
      ! Scale historical I0-nodes to current I_mean (=2*I_bar)
      nodes(i)%I_drive = (x(i) * I0) * (I_mean / I0)
      nodes(i)%phi_drive = p%alpha0_phase
      if (nodes(i)%phi_drive < 0.0_dp) nodes(i)%phi_drive = nodes(i)%phi_drive + TWOPI
      abs_alpha = sqrt(max(nodes(i)%I_drive / max(I_mean, tiny(1.0_dp)), 0.0_dp))
      nodes(i)%re_alpha = abs_alpha * cos(nodes(i)%phi_drive)
      nodes(i)%im_alpha = abs_alpha * sin(nodes(i)%phi_drive)
    end do
  end subroutine build_nodes_exponential_quad

  subroutine gh_nodes_weights(n, n_out, x, w)
    integer, intent(in) :: n
    integer, intent(out) :: n_out
    real(dp), intent(out) :: x(7), w(7)
    x = 0.0_dp; w = 0.0_dp
    select case (n)
    case (3)
      n_out = 3
      x(1) = -1.224744871391589_dp; w(1) = 0.2954089751509193_dp
      x(2) =  0.0_dp;               w(2) = 1.181635900603677_dp
      x(3) =  1.224744871391589_dp; w(3) = 0.2954089751509193_dp
    case (5)
      n_out = 5
      x(1) = -2.020182870456086_dp; w(1) = 0.01995324205904591_dp
      x(2) = -0.9585724646138185_dp; w(2) = 0.3936193231522412_dp
      x(3) =  0.0_dp;               w(3) = 0.9453087204829419_dp
      x(4) =  0.9585724646138185_dp; w(4) = 0.3936193231522412_dp
      x(5) =  2.020182870456086_dp; w(5) = 0.01995324205904591_dp
    case (7)
      n_out = 7
      x(1) = -2.651961356835233_dp; w(1) = 9.717812450995194e-4_dp
      x(2) = -1.673551628767471_dp; w(2) = 5.450729697312067e-2_dp
      x(3) = -0.8162878828589647_dp; w(3) = 0.4256072526101278_dp
      x(4) =  0.0_dp;               w(4) = 0.8102646175568073_dp
      x(5) =  0.8162878828589647_dp; w(5) = 0.4256072526101278_dp
      x(6) =  1.673551628767471_dp; w(6) = 5.450729697312067e-2_dp
      x(7) =  2.651961356835233_dp; w(7) = 9.717812450995194e-4_dp
    case default
      write(*,*) 'ERROR: bsv_gh_order must be 3, 5, or 7; got ', n
      error stop 1
    end select
  end subroutine gh_nodes_weights

  subroutine qlight_parse_propagate_ids(list, ids)
    ! Comma-separated positive integers. Each token must be pure digits only
    ! (optional surrounding spaces). Rejects empty tokens, spaces inside a
    ! token ("1 2,3"), signs, decimals, and scientific notation.
    ! Uses nested IF (not .and.) so buf(i:i) is never evaluated for i>n.
    character(*), intent(in) :: list
    integer, allocatable, intent(out) :: ids(:)
    character(len=len(list)) :: buf, token
    integer :: i, n, ios, v, start, cap, n_ids, k, ntok
    integer, allocatable :: tmp(:)
    logical :: at_delim
    character(1) :: ch

    buf = adjustl(list)
    n = len_trim(buf)
    if (n <= 0) then
      write(*,*) 'ERROR: empty bsv_propagate_ids'
      error stop 1
    end if
    cap = 32
    allocate(tmp(cap))
    n_ids = 0
    start = 1
    do i = 1, n + 1
      at_delim = .false.
      if (i > n) then
        at_delim = .true.
      else
        if (buf(i:i) == ',') at_delim = .true.
      end if
      if (.not. at_delim) cycle

      ! Empty token between delimiters, or trailing/leading comma.
      if (i <= start) then
        write(*,*) 'ERROR: empty token in bsv_propagate_ids: ', trim(list)
        error stop 1
      end if
      token = adjustl(buf(start:i-1))
      ntok = len_trim(token)
      if (ntok <= 0) then
        write(*,*) 'ERROR: empty token in bsv_propagate_ids: ', trim(list)
        error stop 1
      end if
      do k = 1, ntok
        ch = token(k:k)
        if (ch < '0' .or. ch > '9') then
          write(*,*) 'ERROR: bsv_propagate_ids token is not a pure positive integer: ', &
                     trim(token)
          error stop 1
        end if
      end do
      read(token(1:ntok), *, iostat=ios) v
      if (ios /= 0 .or. v <= 0) then
        write(*,*) 'ERROR: invalid bsv_propagate_ids entry near: ', trim(token)
        error stop 1
      end if
      n_ids = n_ids + 1
      if (n_ids > cap) then
        cap = cap * 2
        block
          integer, allocatable :: grow(:)
          allocate(grow(cap))
          grow(1:n_ids-1) = tmp(1:n_ids-1)
          call move_alloc(grow, tmp)
        end block
      end if
      tmp(n_ids) = v
      start = i + 1
    end do
    if (n_ids <= 0) then
      write(*,*) 'ERROR: bsv_propagate_ids parsed zero ids from: ', trim(list)
      error stop 1
    end if
    allocate(ids(n_ids))
    ids = tmp(1:n_ids)
    deallocate(tmp)
  end subroutine qlight_parse_propagate_ids

  subroutine qlight_select_nodes_by_ids(all_nodes, ids, selected)
    type(qlight_node_t), intent(in)  :: all_nodes(:)
    integer, intent(in)              :: ids(:)
    type(qlight_node_t), allocatable, intent(out) :: selected(:)
    integer :: i, j, n_ids, n_all, found

    n_ids = size(ids)
    n_all = size(all_nodes)
    if (n_ids <= 0) then
      write(*,*) 'ERROR: qlight_select_nodes_by_ids called with empty id list'
      error stop 1
    end if
    allocate(selected(n_ids))
    do i = 1, n_ids
      found = 0
      do j = 1, n_all
        if (all_nodes(j)%id == ids(i)) then
          selected(i) = all_nodes(j)
          found = 1
          exit
        end if
      end do
      if (found == 0) then
        write(*,*) 'ERROR: propagate id not found in manifest: ', ids(i)
        error stop 1
      end if
    end do
    do i = 1, n_ids - 1
      do j = i + 1, n_ids
        if (ids(i) == ids(j)) then
          write(*,*) 'ERROR: duplicate id in bsv_propagate_ids: ', ids(i)
          error stop 1
        end if
      end do
    end do
  end subroutine qlight_select_nodes_by_ids

end module mod_quantum_light
