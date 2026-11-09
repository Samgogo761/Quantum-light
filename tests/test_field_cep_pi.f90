!===============================================================================
! test_field_cep_pi
!-------------------------------------------------------------------------------
! Production-path check: generate_field_sample antipode cancellation.
!
!   max_t || E(phi) + E(phi+pi) || / E_max  < 1e-14
!
! Uses the real mod_laser.generate_field_sample (not a Python replica).
! Build via: make test-field-cep  (needs mod_params + mod_laser only)
!===============================================================================
program test_field_cep_pi
  use mod_params
  use mod_laser
  implicit none

  real(dp), parameter :: TOL = 1.0e-14_dp
  real(dp) :: E_peak, phi_a, phi_b
  real(dp) :: max_sum, e_ref, rel
  real(dp), allocatable :: Et_a(:,:)
  integer :: it
  logical :: pass

  print '(A)', '================================================================'
  print '(A)', '  mod_laser : CEP-pi field antipode unit test'
  print '(A)', '================================================================'

  call init_minimal_laser_params()

  ! Representative strong-node drive (production-scale).
  E_peak = sqrt(5.95984289e11_dp * Wcm2_to_au)
  phi_a  = 4.7123889803846897_dp   ! ~ 3*pi/2
  phi_b  = modulo(phi_a + PI, TWOPI)

  call generate_field_sample(E_peak, phi_a)
  allocate(Et_a(nt, 3))
  Et_a = Et_vec

  call generate_field_sample(E_peak, phi_b)

  max_sum = 0.0_dp
  e_ref = 0.0_dp
  do it = 1, nt
    e_ref = max(e_ref, sqrt(Et_a(it,1)**2 + Et_a(it,2)**2 + Et_a(it,3)**2))
    e_ref = max(e_ref, sqrt(Et_vec(it,1)**2 + Et_vec(it,2)**2 + Et_vec(it,3)**2))
    max_sum = max(max_sum, sqrt( &
      (Et_a(it,1) + Et_vec(it,1))**2 + &
      (Et_a(it,2) + Et_vec(it,2))**2 + &
      (Et_a(it,3) + Et_vec(it,3))**2 ))
  end do
  rel = max_sum / max(e_ref, 1.0e-300_dp)
  pass = (rel < TOL)

  print '(A,ES16.8)', '  E_peak        = ', E_peak
  print '(A,ES16.8)', '  phi_a         = ', phi_a
  print '(A,ES16.8)', '  phi_b         = ', phi_b
  print '(A,I0)',     '  nt            = ', nt
  print '(A,ES16.8)', '  max||Esum||   = ', max_sum
  print '(A,ES16.8)', '  E_max         = ', e_ref
  print '(A,ES16.8)', '  rel           = ', rel
  print '(A,ES16.8)', '  tol           = ', TOL
  if (pass) then
    print '(A)', '  RESULT: PASS'
  else
    print '(A)', '  RESULT: FAIL'
    error stop 1
  end if

  ! Second check: tiny phase digitization like legacy 16.8e must FAIL the 1e-14 gate.
  phi_b = phi_a + PI + 3.6e-9_dp
  call generate_field_sample(E_peak, phi_a)
  Et_a = Et_vec
  call generate_field_sample(E_peak, phi_b)
  max_sum = 0.0_dp
  e_ref = 0.0_dp
  do it = 1, nt
    e_ref = max(e_ref, sqrt(Et_a(it,1)**2 + Et_a(it,2)**2 + Et_a(it,3)**2))
    max_sum = max(max_sum, sqrt( &
      (Et_a(it,1) + Et_vec(it,1))**2 + &
      (Et_a(it,2) + Et_vec(it,2))**2 + &
      (Et_a(it,3) + Et_vec(it,3))**2 ))
  end do
  rel = max_sum / max(e_ref, 1.0e-300_dp)
  print '(A)', '----------------------------------------------------------------'
  print '(A)', '  [diag] legacy-scale dphi=3.6e-9 residual (expect ~dphi, not <1e-14)'
  print '(A,ES16.8)', '  rel_legacy_dphi = ', rel
  if (rel < 1.0e-12_dp) then
    print '(A)', '  WARNING: legacy dphi residual unexpectedly tiny'
  end if
  print '(A)', '================================================================'
  deallocate(Et_a)

contains

  subroutine init_minimal_laser_params()
    ! Mirror the production laser + timestep block used by full112 GH3.
    wvl_nm = 3200.0_dp
    intensity_Wcm2 = 2.0e11_dp
    theta_deg = 0.0_dp
    phi_cep_deg = 90.0_dp
    ncyc = 4.0_dp
    env_type = 2
    ellipticity = 0.0_dp
    delta_phase_deg = 0.0_dp
    dt = 0.35_dp
    dual_color = .false.
    use_external_A = .false.

    omega0    = TWOPI * c_au / (wvl_nm * nm_to_bohr)
    E0        = sqrt(intensity_Wcm2 * Wcm2_to_au)
    A0        = E0 / omega0
    T_cycle   = TWOPI / omega0
    T_total_1 = ncyc * T_cycle
    T_total   = T_total_1
    theta     = theta_deg * PI / 180.0_dp
    phi_cep   = phi_cep_deg * PI / 180.0_dp
    delta_phase = delta_phase_deg * PI / 180.0_dp
    pol_vec = [cos(theta), sin(theta), 0.0_dp]
    nt = ceiling(T_total / dt) + 1
    if (nt < 1) nt = 1
  end subroutine init_minimal_laser_params

end program test_field_cep_pi
