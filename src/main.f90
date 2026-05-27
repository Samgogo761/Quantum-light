program hhg_sbe_solver
  use iso_fortran_env, only: int64
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  use mod_sbe
  use mod_current
  use mod_hhg
  use mod_quantum_light, only: qlight_params_t, qlight_init, qlight_sample_bsv
  use mod_berry
  implicit none

  real(dp) :: t_start, t_end
  real(dp), allocatable :: hhg_x(:), hhg_y(:), hhg_tot(:)
  integer :: n_omega
  character(256) :: input_file

  ! BSV variables
  type(qlight_params_t) :: qp
  real(dp), allocatable :: Jt_sample(:,:), hhg_accum(:), hhg_accum2(:)
  real(dp), allocatable :: hhg_x_s(:), hhg_y_s(:), hhg_tot_s(:)
  real(dp) :: I_sample, phi_sample, E_peak_sample
  integer :: isamp, n_omega_s, n_omega_sample

  t_start = wall_time_seconds()

  ! === 1. Read input ===
  call resolve_input_file(input_file)
  write(*,'(A)') 'Reading input from '//trim(input_file)
  call read_input(trim(input_file))

  ! === 2. Read Wannier90 data ===
  if (len_trim(wannier_tb_file) > 0) then
    call read_tb_file(wannier_tb_file)
  else
    call read_hr_file(wannier_hr_file)
    if (len_trim(wannier_r_file) > 0) call read_r_file(wannier_r_file)
  end if

  ! === 3. Build k-grid and band structure ===
  call setup_kgrid()
  call compute_valley_assignment()
  call compute_band_structure()

  if (run_pcenter_check) then
    call print_params()
    call write_bands("bands.dat")
    call diagnose_pcenter_velocity(trim(pcenter_summary_file), trim(pcenter_kresolved_file))
    if (stop_after_diagnostics) then
      t_end = wall_time_seconds()
      write(*,'(A,F10.2,A)') 'Total wall time: ', t_end - t_start, ' seconds'
      stop
    end if
  end if

  select case (trim(gauge_method))
  case ('vg')
    call precompute_projected_matrices()
  case ('matrix_vg')
    call precompute_matrix_vg_matrices()
  case ('lg', 'lg_cov', 'houston_lg')
    call precompute_lg_matrices()
  case default
    write(*,*) 'ERROR: gauge_method must be "vg", "matrix_vg", "lg", or "lg_cov", got: ', &
               trim(gauge_method)
    error stop 1
  end select
  call diagnose_tb_quality()
  if (allocated(HR_proj)) call compute_berry_curvature()
  call print_params()
  call write_bands("bands.dat")
  if (allocated(HR_proj)) call write_berry_curvature("berry_curvature.dat")

  ! === 4. Branch: classical or BSV ===
  if (.not. bsv_enabled) then

    ! --- Classical single-trajectory path ---
    call generate_field()
    call init_density_matrix()

    write(*,'(A)') 'Starting classical time evolution...'
    call propagate()
    write(*,'(A)') 'Time evolution complete.'
    write(*,'(A,ES12.4,A,2ES12.4,A)') 'Initial current |J(t=0)|: ', &
      sqrt(Jt(1,1)**2 + Jt(1,2)**2), '  (Jx,Jy)=(', Jt(1,1), Jt(1,2), ')'

    call write_current("Jt.dat", Jt, nt, dt)
    call write_current_decomposed("Jt_decomposed.dat", Jt, Jt_intra, Jt_inter, nt, dt)
    if (allocated(valley_id)) then
      call write_valley_current("Jt_valley.dat", Jt_K, Jt_Kp, nt, dt)
    end if
    call compute_hhg_spectrum(Jt, nt, dt, omega0, hhg_x, hhg_y, hhg_tot, n_omega)
    call write_hhg("HHG.dat", hhg_x, hhg_y, hhg_tot, n_omega, nt, dt, omega0)
    deallocate(hhg_x, hhg_y, hhg_tot)

  else

    ! --- BSV Monte Carlo ensemble path ---
    if (bsv_n_samples <= 0) then
      write(*,*) 'ERROR: bsv_n_samples must be positive when bsv_enabled = .true.'
      error stop 1
    end if

    write(*,'(A,I0,A)') 'Starting BSV ensemble with ', bsv_n_samples, ' trajectories...'
    write(*,'(A,ES12.4,A)') '  BSV I_bar scale        : ', bsv_mean_intensity, ' W/cm^2'
    write(*,'(A,ES12.4,A)') '  BSV sampled mean <I>   : ', 2.0_dp*bsv_mean_intensity, ' W/cm^2'

    qp%enabled   = .true.
    qp%I_bar     = bsv_mean_intensity
    qp%n_samples = bsv_n_samples
    qp%seed      = bsv_seed
    call qlight_init(qp)

    n_omega_s = nt / 2 + 1
    allocate(Jt_sample(nt, 2))
    allocate(hhg_accum(n_omega_s), hhg_accum2(n_omega_s))
    hhg_accum  = 0.0_dp
    hhg_accum2 = 0.0_dp

    do isamp = 1, bsv_n_samples
      call qlight_sample_bsv(qp, I_sample, phi_sample)

      E_peak_sample = sqrt(I_sample * Wcm2_to_au)
      call generate_field_sample(E_peak_sample, phi_sample)
      call run_single_trajectory(Jt_sample)

      call compute_hhg_spectrum(Jt_sample, nt, dt, omega0, &
                                 hhg_x_s, hhg_y_s, hhg_tot_s, n_omega_sample)

      if (n_omega_sample /= n_omega_s) then
        write(*,*) 'ERROR: inconsistent HHG frequency grid in BSV ensemble.'
        error stop 1
      end if

      hhg_accum  = hhg_accum  + hhg_tot_s
      hhg_accum2 = hhg_accum2 + hhg_tot_s**2
      deallocate(hhg_x_s, hhg_y_s, hhg_tot_s)

      if (mod(isamp, 50) == 0) then
        write(*,'(A,I0,A,I0)') '  BSV sample ', isamp, ' / ', bsv_n_samples
      end if
    end do

    hhg_accum  = hhg_accum  / real(bsv_n_samples, dp)
    hhg_accum2 = hhg_accum2 / real(bsv_n_samples, dp)
    hhg_accum2 = sqrt(max(hhg_accum2 - hhg_accum**2, 0.0_dp) / real(bsv_n_samples, dp))

    call write_bsv_hhg("HHG_bsv.dat", hhg_accum, hhg_accum2, n_omega_s, &
                        bsv_n_samples, nt, dt, omega0)

    deallocate(Jt_sample, hhg_accum, hhg_accum2)
    write(*,'(A)') 'BSV ensemble complete.'
  end if

  t_end = wall_time_seconds()
  write(*,'(A,F10.2,A)') 'Total wall time: ', t_end - t_start, ' seconds'

contains

  real(dp) function wall_time_seconds()
    integer(int64) :: count, rate

    call system_clock(count, rate)
    wall_time_seconds = real(count, dp) / real(rate, dp)
  end function wall_time_seconds

  subroutine resolve_input_file(path)
    character(*), intent(out) :: path
    logical :: exists

    if (command_argument_count() >= 1) then
      call get_command_argument(1, path)
      inquire(file=trim(path), exist=exists)
      if (.not. exists) then
        write(*,*) 'ERROR: input file from command line not found: ', trim(path)
        error stop 1
      end if
      return
    end if

    inquire(file='input.nml', exist=exists)
    if (exists) then
      path = 'input.nml'
      return
    end if

    inquire(file='input/input.nml', exist=exists)
    if (exists) then
      path = 'input/input.nml'
      return
    end if

    write(*,*) 'ERROR: cannot find input.nml in current directory or input/input.nml.'
    error stop 1
  end subroutine resolve_input_file

  subroutine write_bsv_hhg(filename, hhg_avg, hhg_stderr, nw, n_samp, nt_in, dt_in, omega0_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: hhg_avg(:), hhg_stderr(:)
    integer,      intent(in) :: nw, n_samp, nt_in
    real(dp),     intent(in) :: dt_in, omega0_in
    integer  :: u, iw
    real(dp) :: domega, omega_n, h_order

    domega = TWOPI / (real(nt_in, dp) * dt_in)
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A,I0)') '# n_samples = ', n_samp
    write(u, '(A)') '# harmonic_order  omega(a.u.)  HHG_bsv_avg  HHG_bsv_stderr'
    do iw = 1, nw
      omega_n = real(iw - 1, dp) * domega
      h_order = omega_n / omega0_in
      write(u, '(F10.4, 3ES16.8)') h_order, omega_n, hhg_avg(iw), hhg_stderr(iw)
    end do
    close(u)
  end subroutine write_bsv_hhg

end program hhg_sbe_solver
