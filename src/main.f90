program hhg_sbe_solver
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  use mod_sbe
  use mod_current
  use mod_hhg
  use mod_quantum_light, only: qlight_params_t, qlight_init, qlight_sample_bsv
  implicit none

  real(dp) :: t_start, t_end
  real(dp), allocatable :: hhg_x(:), hhg_y(:), hhg_tot(:)
  integer :: n_omega
  character(256) :: input_file

  ! BSV variables
  type(qlight_params_t) :: qp
  real(dp), allocatable :: Jt_sample(:,:), hhg_accum(:)
  real(dp), allocatable :: hhg_x_s(:), hhg_y_s(:), hhg_tot_s(:)
  real(dp) :: I_sample, phi_sample, E_peak_sample
  integer :: isamp, n_omega_s

  call cpu_time(t_start)

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
  call compute_band_structure()
  call precompute_projected_matrices()
  call print_params()
  call write_bands("bands.dat")

  ! === 4. Branch: classical or BSV ===
  if (.not. bsv_enabled) then

    ! --- Classical single-trajectory path ---
    call generate_field()
    call init_density_matrix()

    write(*,'(A)') 'Starting classical time evolution...'
    call propagate()
    write(*,'(A)') 'Time evolution complete.'

    call write_current("Jt.dat", Jt, nt, dt)
    call compute_hhg_spectrum(Jt, nt, dt, omega0, hhg_x, hhg_y, hhg_tot, n_omega)
    call write_hhg("HHG.dat", hhg_x, hhg_y, hhg_tot, n_omega, nt, dt, omega0)
    deallocate(hhg_x, hhg_y, hhg_tot)

  else

    ! --- BSV Monte Carlo ensemble path ---
    write(*,'(A,I0,A)') 'Starting BSV ensemble with ', bsv_n_samples, ' trajectories...'

    qp%enabled   = .true.
    qp%I_bar     = bsv_mean_intensity
    qp%n_samples = bsv_n_samples
    qp%seed      = bsv_seed
    call qlight_init(qp)

    allocate(Jt_sample(nt, 2))

    do isamp = 1, bsv_n_samples
      call qlight_sample_bsv(qp, I_sample, phi_sample)

      E_peak_sample = sqrt(I_sample * Wcm2_to_au)
      call generate_field_sample(E_peak_sample, phi_sample)
      call run_single_trajectory(Jt_sample)

      call compute_hhg_spectrum(Jt_sample, nt, dt, omega0, &
                                 hhg_x_s, hhg_y_s, hhg_tot_s, n_omega_s)

      if (.not. allocated(hhg_accum)) then
        allocate(hhg_accum(n_omega_s))
        hhg_accum = 0.0_dp
      end if

      hhg_accum = hhg_accum + hhg_tot_s
      deallocate(hhg_x_s, hhg_y_s, hhg_tot_s)

      if (mod(isamp, 50) == 0) then
        write(*,'(A,I0,A,I0)') '  BSV sample ', isamp, ' / ', bsv_n_samples
      end if
    end do

    hhg_accum = hhg_accum / real(bsv_n_samples, dp)

    call write_bsv_hhg("HHG_bsv.dat", hhg_accum, n_omega_s, nt, dt, omega0)

    deallocate(Jt_sample, hhg_accum)
    write(*,'(A)') 'BSV ensemble complete.'
  end if

  call cpu_time(t_end)
  write(*,'(A,F10.2,A)') 'Total wall time: ', t_end - t_start, ' seconds'

contains

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

  subroutine write_bsv_hhg(filename, hhg_avg, nw, nt_in, dt_in, omega0_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: hhg_avg(:)
    integer,      intent(in) :: nw, nt_in
    real(dp),     intent(in) :: dt_in, omega0_in
    integer  :: u, iw
    real(dp) :: domega, omega_n, h_order

    domega = TWOPI / (real(nt_in, dp) * dt_in)
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# harmonic_order  omega(a.u.)  HHG_bsv_avg'
    do iw = 1, nw
      omega_n = real(iw - 1, dp) * domega
      h_order = omega_n / omega0_in
      write(u, '(F10.4, 2ES16.8)') h_order, omega_n, hhg_avg(iw)
    end do
    close(u)
  end subroutine write_bsv_hhg

end program hhg_sbe_solver
