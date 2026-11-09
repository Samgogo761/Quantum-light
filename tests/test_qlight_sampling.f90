!===============================================================================
! test_qlight_sampling
!-------------------------------------------------------------------------------
! Stand-alone unit tests for mod_quantum_light (layer A0):
!
!   (1) random_phase_exponential historical benchmark
!         <I>    -> 2 * I_bar
!         Var(I) -> (2 * I_bar)^2
!         phi histogram flat
!   (2) Q0 moment gates for coherent / squeezed_vacuum
!
! Build:
!     cd tests/
!     make
!     ./test_qlight_sampling
!===============================================================================
program test_qlight_sampling
  use mod_quantum_light
  implicit none

  integer,  parameter :: N       = 500000
  integer,  parameter :: N_Q0    = 200000
  integer,  parameter :: N_BINS  = 50
  real(dp), parameter :: I_BAR   = 2.0e11_dp
  real(dp), parameter :: TOL_MEAN = 5.0e-3_dp
  real(dp), parameter :: TOL_VAR  = 1.0e-2_dp
  real(dp), parameter :: TOL_PHI  = 5.0e-3_dp

  type(qlight_params_t) :: qp
  real(dp) :: I_i, phi_i
  real(dp) :: sum_I, sum_I2, mean_I, var_I
  real(dp) :: expected_mean, expected_var
  real(dp) :: rel_mean, rel_var
  real(dp) :: I_max, I_min
  integer  :: phi_hist(N_BINS)
  integer  :: i, ibin, nan_inf, n_tail_10x
  logical  :: pass_exp, pass_q0_coh, pass_q0_sv, pass_all

  print '(A)', '================================================================'
  print '(A)', '  mod_quantum_light : A0 sampler unit tests'
  print '(A)', '================================================================'

  ! ---- (1) random_phase_exponential -----------------------------------------
  qp%enabled    = .true.
  qp%I_bar      = I_BAR
  qp%n_samples  = N
  qp%seed       = 12345
  qp%state_type = 'random_phase_exponential'
  call qlight_init(qp)

  sum_I = 0.0_dp; sum_I2 = 0.0_dp
  I_max = -huge(1.0_dp); I_min = huge(1.0_dp)
  phi_hist = 0; nan_inf = 0; n_tail_10x = 0

  do i = 1, N
    call qlight_sample_random_phase_exponential(qp, I_i, phi_i)
    if (I_i /= I_i .or. I_i > huge(1.0_dp)) then
      nan_inf = nan_inf + 1
      cycle
    end if
    sum_I  = sum_I  + I_i
    sum_I2 = sum_I2 + I_i * I_i
    if (I_i > I_max) I_max = I_i
    if (I_i < I_min) I_min = I_i
    if (I_i > 10.0_dp * I_BAR) n_tail_10x = n_tail_10x + 1
    ibin = int(phi_i / TWOPI * real(N_BINS, dp)) + 1
    if (ibin < 1) ibin = 1
    if (ibin > N_BINS) ibin = N_BINS
    phi_hist(ibin) = phi_hist(ibin) + 1
  end do

  mean_I = sum_I / real(N, dp)
  var_I  = sum_I2 / real(N, dp) - mean_I * mean_I
  expected_mean = 2.0_dp * I_BAR
  expected_var  = (2.0_dp * I_BAR)**2
  rel_mean = abs(mean_I - expected_mean) / expected_mean
  rel_var  = abs(var_I  - expected_var ) / expected_var

  print '(A)', '----------------------------------------------------------------'
  print '(A)', '  [1] random_phase_exponential (legacy circular benchmark)'
  print '(A,ES14.6,A,ES14.6)', '  <I>     : ', mean_I, '  expected ', expected_mean
  print '(A,ES14.6,A,ES14.6)', '  Var(I)  : ', var_I , '  expected ', expected_var
  print '(A,ES12.4)', '  rel <I> : ', rel_mean
  print '(A,ES12.4)', '  rel Var : ', rel_var
  print '(A,I0)', '  NaN/Inf : ', nan_inf
  print '(A,I0)', '  I>10 Ibar samples : ', n_tail_10x

  pass_exp = .true.
  block
    real(dp) :: expected_count, dev, worst
    integer :: k
    expected_count = real(N, dp) / real(N_BINS, dp)
    worst = 0.0_dp
    do k = 1, N_BINS
      dev = abs(real(phi_hist(k), dp) - expected_count) / expected_count
      if (dev > worst) worst = dev
    end do
    print '(A,ES12.4)', '  max phi bin rel dev : ', worst
    if (nan_inf > 0) pass_exp = .false.
    if (rel_mean > TOL_MEAN) pass_exp = .false.
    if (rel_var  > TOL_VAR)  pass_exp = .false.
    if (worst > TOL_PHI * 10.0_dp) pass_exp = .false.
    if (n_tail_10x < 1) pass_exp = .false.
  end block
  if (pass_exp) then
    print '(A)', '  RESULT[1] : PASS'
  else
    print '(A)', '  RESULT[1] : FAIL'
  end if

  ! ---- (2a) Q0 coherent -----------------------------------------------------
  print '(A)', '----------------------------------------------------------------'
  print '(A)', '  [2a] Q0 coherent'
  qp%state_type = 'coherent'
  qp%alpha0_abs = 3.0_dp
  qp%alpha0_phase = 0.4_dp
  qp%seed = 99
  call qlight_init(qp)
  call qlight_validate_Q0_report(qp, N_Q0, 'q0_coherent_report.txt', pass_q0_coh)
  if (pass_q0_coh) then
    print '(A)', '  RESULT[2a]: PASS  (see q0_coherent_report.txt)'
  else
    print '(A)', '  RESULT[2a]: FAIL  (see q0_coherent_report.txt)'
  end if

  ! ---- (2b) Q0 squeezed vacuum ----------------------------------------------
  print '(A)', '----------------------------------------------------------------'
  print '(A)', '  [2b] Q0 squeezed_vacuum'
  qp%state_type = 'squeezed_vacuum'
  qp%squeeze_r = 0.8_dp
  qp%squeeze_theta = 0.6_dp
  qp%alpha0_abs = 0.0_dp
  qp%seed = 77
  call qlight_init(qp)
  call qlight_validate_Q0_report(qp, N_Q0, 'q0_sv_report.txt', pass_q0_sv)
  if (pass_q0_sv) then
    print '(A)', '  RESULT[2b]: PASS  (see q0_sv_report.txt)'
  else
    print '(A)', '  RESULT[2b]: FAIL  (see q0_sv_report.txt)'
  end if

  ! ---- legacy alias still works ---------------------------------------------
  print '(A)', '----------------------------------------------------------------'
  print '(A)', '  [3] legacy qlight_sample_bsv alias'
  qp%state_type = 'bsv'
  qp%I_bar = I_BAR
  qp%seed = 1
  call qlight_init(qp)
  call qlight_sample_bsv(qp, I_i, phi_i)
  print '(A,ES12.4,A,ES12.4)', '  sample I,phi : ', I_i, '  ', phi_i

  pass_all = pass_exp .and. pass_q0_coh .and. pass_q0_sv
  print '(A)', '================================================================'
  if (pass_all) then
    print '(A)', '  OVERALL : PASS'
    print '(A)', '================================================================'
    stop 0
  else
    print '(A)', '  OVERALL : FAIL'
    print '(A)', '================================================================'
    stop 1
  end if

end program test_qlight_sampling
