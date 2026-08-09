program hhg_sbe_solver
  use iso_fortran_env, only: int64
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  use mod_sbe
  use mod_current
  use mod_hhg
  use mod_quantum_light, only: qlight_params_t, qlight_node_t, qlight_init, &
                               qlight_normalize_state_type, qlight_normalize_sampling_mode, &
                               qlight_build_nodes, qlight_write_nodes_manifest, &
                               qlight_validate_nodes_moments, qlight_validate_nodes_integrity, &
                               qlight_parse_propagate_ids, qlight_select_nodes_by_ids
  use mod_berry
  use mod_geometry
  use mod_spin
  implicit none

  real(dp) :: t_start, t_end
  real(dp), allocatable :: hhg_x(:), hhg_y(:), hhg_tot(:)
  integer :: n_omega
  character(256) :: input_file

    type(qlight_params_t) :: qp
    type(qlight_node_t), allocatable :: nodes_all(:), nodes(:)
    integer, allocatable :: prop_ids(:)
    logical :: chunk_mode, write_ics_cs
  real(dp), allocatable :: Jt_sample(:,:), hhg_x_s(:), hhg_y_s(:), hhg_tot_s(:)
  real(dp), allocatable :: ics_tot(:), cs_tot(:), var_tot(:)
  complex(dp), allocatable :: Jw_x(:), Jw_y(:), sum_Jw_x(:), sum_Jw_y(:)
  real(dp) :: E_peak_sample, spectrum_scale, w, omega_n, h_order, domega
  integer :: isamp, n_omega_s, n_omega_sample, n_nodes
  integer, allocatable :: harm_orders(:)
    integer :: n_harm, ih, iw_h, u_modes, uid_chunk
  character(64) :: state_label, mode_label
  character(256) :: nodes_path, cplx_name
  logical :: nodes_ok
  character(16) :: tag

  t_start = wall_time_seconds()

  call resolve_input_file(input_file)
  write(*,'(A)') 'Reading input from '//trim(input_file)
  call read_input(trim(input_file))

  if (len_trim(wannier_tb_file) > 0) then
    call read_tb_file(wannier_tb_file)
  else
    call read_hr_file(wannier_hr_file)
    if (len_trim(wannier_r_file) > 0) call read_r_file(wannier_r_file)
  end if

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

  if (save_geometry .and. allocated(Pk_eq)) then
    call compute_quantum_geometry()
    call write_quantum_geometry("quantum_geometry.dat")
  end if

  if (spin_current .and. allocated(Pk_eq)) then
    call setup_spin_operator(Pk_eq)
  else if (spin_current) then
    write(*,'(A)') '  NOTE: spin_current requires the lg_cov/matrix_vg path (Pk_eq); skipped.'
  end if

  if (.not. bsv_enabled) then
    call generate_field()
    call write_field("Et.dat")
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

    call compute_complex_current_spectrum(Jt, nt, dt, omega0, &
                                          Jw_x, Jw_y, n_omega, spectrum_scale)
    call intensities_from_complex(Jw_x, Jw_y, spectrum_scale, hhg_x, hhg_y, hhg_tot)
    call write_hhg("HHG.dat", hhg_x, hhg_y, hhg_tot, n_omega, nt, dt, omega0)
    if (save_complex_hhg) then
      call write_hhg_complex("HHG_complex.dat", Jw_x, Jw_y, spectrum_scale, &
                             n_omega, nt, dt, omega0)
      write(*,'(A)') 'Wrote HHG_complex.dat (classical trajectory complex amplitudes).'
    end if
    deallocate(hhg_x, hhg_y, hhg_tot, Jw_x, Jw_y)

    if (spin_current .and. spin_ready) then
      call write_spin_current("Jt_spin.dat", nt, dt)
      call compute_hhg_spectrum(Jt_spin, nt, dt, omega0, hhg_x, hhg_y, hhg_tot, n_omega)
      call write_hhg("HHG_spin.dat", hhg_x, hhg_y, hhg_tot, n_omega, nt, dt, omega0)
      deallocate(hhg_x, hhg_y, hhg_tot)
    end if

  else
    state_label = qlight_normalize_state_type(bsv_state_type)
    mode_label  = qlight_normalize_sampling_mode(bsv_sampling_mode)

    write(*,'(A)') 'Starting quantum-light ensemble (layer A)...'
    write(*,'(A,A)')    '  state_type             : ', trim(state_label)
    write(*,'(A,A)')    '  sampling_mode          : ', trim(mode_label)
    write(*,'(A,ES12.4,A)') '  I_bar scale            : ', bsv_mean_intensity, ' W/cm^2'
    write(*,'(A,ES12.4,A)') '  target drive mean <I>  : ', 2.0_dp*bsv_mean_intensity, ' W/cm^2'
    write(*,'(A)') '  NOTE: equal-mean-intensity mapping I=I_mean*|α|^2/E_Q[|α|^2];'
    write(*,'(A)') '        NOT experimental BSV pulse-energy calibration.'
    if (trim(state_label) == 'random_phase_exponential') then
      write(*,'(A)') '  NOTE: RPE is circular/exponential benchmark, NOT fixed-angle SV.'
    end if

    qp%enabled   = .true.
    qp%I_bar     = bsv_mean_intensity
    qp%n_samples = bsv_n_samples
    qp%seed      = bsv_seed
    qp%state_type = state_label
    qp%squeeze_r = bsv_squeeze_r
    qp%squeeze_theta = bsv_squeeze_theta_deg * PI / 180.0_dp
    qp%alpha0_abs = bsv_alpha0_abs
    qp%alpha0_phase = bsv_alpha0_phase_deg * PI / 180.0_dp
    qp%thermal_nbar = bsv_thermal_nbar
    call qlight_init(qp)

    if (len_trim(bsv_nodes_file) > 0) then
      nodes_path = bsv_nodes_file
    else
      nodes_path = 'nodes_manifest.dat'
    end if

    call qlight_build_nodes(qp, mode_label, bsv_gh_order, nodes_path, nodes_all)
    n_nodes = size(nodes_all)
    write(*,'(A,I0,A)') '  n_nodes (manifest)     : ', n_nodes, ' (weighted)'

    if (bsv_write_nodes .or. trim(mode_label) /= 'from_file') then
      call qlight_write_nodes_manifest(nodes_path, nodes_all, state_label, mode_label)
      write(*,'(A,A)') '  wrote nodes manifest   : ', trim(nodes_path)
    end if
    call qlight_validate_nodes_integrity(qp, nodes_all, 'nodes_integrity_check.txt', nodes_ok)
    if (.not. nodes_ok) then
      write(*,*) 'ERROR: nodes integrity check FAILED (see nodes_integrity_check.txt).'
      write(*,*) '  Production / from_file path refuses to continue with illegal manifests.'
      error stop 1
    end if
    call qlight_validate_nodes_moments(qp, nodes_all, 'nodes_moment_check.txt', nodes_ok)
    if (.not. nodes_ok) then
      write(*,*) 'ERROR: nodes moment check FAILED (see nodes_moment_check.txt).'
      write(*,*) '  Hard stop enabled so GH/from_file production cannot silently continue.'
      error stop 1
    end if
    write(*,'(A)') '  nodes integrity+moment   : PASS (see nodes_*_check.txt)'

    chunk_mode = .false.
    if (len_trim(bsv_propagate_ids) > 0) then
      call qlight_parse_propagate_ids(trim(bsv_propagate_ids), prop_ids)
      call qlight_select_nodes_by_ids(nodes_all, prop_ids, nodes)
      chunk_mode = (size(nodes) < size(nodes_all))
      n_nodes = size(nodes)
      write(*,'(A,I0,A,I0,A)') '  propagate subset       : ', n_nodes, ' / ', size(nodes_all), ' nodes'
      if (chunk_mode) then
        write(*,'(A)') '  CHUNK MODE: partial propagation; ICS/CS deferred to merge step.'
      end if
    else
      allocate(nodes(n_nodes))
      nodes = nodes_all
    end if
    write_ics_cs = bsv_save_ics_cs .and. .not. chunk_mode

    call parse_harmonic_list(bsv_harmonics, harm_orders, n_harm)
    n_omega_s = nt / 2 + 1
    allocate(Jt_sample(nt, 2))
    allocate(sum_Jw_x(n_omega_s), sum_Jw_y(n_omega_s))
    allocate(ics_tot(n_omega_s), cs_tot(n_omega_s), var_tot(n_omega_s))
    sum_Jw_x = (0.0_dp, 0.0_dp)
    sum_Jw_y = (0.0_dp, 0.0_dp)
    ics_tot = 0.0_dp

    open(newunit=u_modes, file='HHG_nodes_modes.dat', status='replace', action='write')
    write(u_modes,'(A)') '# Per-node complex amplitudes at target harmonics (layer A)'
    write(u_modes,'(A,A)') '# state_type = ', trim(state_label)
    write(u_modes,'(A,A)') '# sampling_mode = ', trim(mode_label)
    write(u_modes,'(A)') '# id weight I phi order ReJx ImJx ReJy ImJy |J|^2_scaled'
    write(u_modes,'(A)') '# WARNING: classical trajectory amplitudes, NOT quantum b_n.'

    do isamp = 1, n_nodes
      w = nodes(isamp)%weight
      E_peak_sample = sqrt(nodes(isamp)%I_drive * Wcm2_to_au)
      call generate_field_sample(E_peak_sample, nodes(isamp)%phi_drive)
      call run_single_trajectory(Jt_sample)

      call compute_complex_current_spectrum(Jt_sample, nt, dt, omega0, &
                                            Jw_x, Jw_y, n_omega_sample, spectrum_scale)
      if (n_omega_sample /= n_omega_s) then
        write(*,*) 'ERROR: inconsistent HHG frequency grid in quantum-light ensemble.'
        error stop 1
      end if
      call intensities_from_complex(Jw_x, Jw_y, spectrum_scale, hhg_x_s, hhg_y_s, hhg_tot_s)

      ics_tot = ics_tot + w * hhg_tot_s
      sum_Jw_x = sum_Jw_x + w * Jw_x
      sum_Jw_y = sum_Jw_y + w * Jw_y

      do ih = 1, n_harm
        iw_h = harmonic_fft_index(nt, dt, omega0, harm_orders(ih))
        ! ES25.17: Jo = [J(+N)-J(-N)]/2 is a complex difference; weak H10
        ! channels need ~17 significant digits, not ~9 (ES16.8).
        write(u_modes,'(I8,3ES25.17,I6,5ES25.17)') nodes(isamp)%id, w, &
          nodes(isamp)%I_drive, nodes(isamp)%phi_drive, harm_orders(ih), &
          real(Jw_x(iw_h), dp), aimag(Jw_x(iw_h)), &
          real(Jw_y(iw_h), dp), aimag(Jw_y(iw_h)), &
          spectrum_scale * (abs(Jw_x(iw_h))**2 + abs(Jw_y(iw_h))**2)
      end do

      if (bsv_save_complex) then
        if (bsv_save_all_complex .or. isamp == 1) then
          write(tag, '(I4.4)') nodes(isamp)%id
          cplx_name = 'HHG_complex_node_'//trim(tag)//'.dat'
          call write_hhg_complex(trim(cplx_name), Jw_x, Jw_y, spectrum_scale, &
                                 n_omega_s, nt, dt, omega0)
        end if
      end if

      deallocate(hhg_x_s, hhg_y_s, hhg_tot_s, Jw_x, Jw_y)
      if (mod(isamp, max(1, n_nodes/10)) == 0 .or. isamp == n_nodes) then
        write(*,'(A,I0,A,I0)') '  node ', isamp, ' / ', n_nodes
      end if
    end do
    close(u_modes)

    cs_tot = spectrum_scale * abs(sum_Jw_x)**2 + spectrum_scale * abs(sum_Jw_y)**2
    var_tot = ics_tot - cs_tot
    if (any(var_tot < -1.0e-12_dp * max(ics_tot, 1.0e-300_dp))) then
      write(*,'(A)') '  WARNING: classical_trajectory_variance slightly negative (roundoff).'
      where (var_tot < 0.0_dp) var_tot = 0.0_dp
    end if

    allocate(hhg_x(n_omega_s)); hhg_x = 0.0_dp
    if (write_ics_cs) then
      call write_bsv_hhg("HHG_bsv.dat", ics_tot, hhg_x, n_omega_s, &
                          n_nodes, nt, dt, omega0, state_label)
    end if
    deallocate(hhg_x)
    if (write_ics_cs) then
      call write_hhg_ics_cs("HHG_ics_cs.dat", ics_tot, cs_tot, var_tot, &
                            n_omega_s, nt, dt, omega0, n_nodes, state_label)
      write(*,'(A)') 'Wrote HHG_ics_cs.dat (weighted ICS/CS/classical_trajectory_variance).'
      write(*,'(A)') '  WARNING: variance is NOT quantum g^(2)/squeezing.'
    else if (chunk_mode) then
      open(newunit=uid_chunk, file='chunk_info.txt', status='replace', action='write')
      write(uid_chunk, '(A)') '# partial node-chunk propagation (ICS/CS not final)'
      write(uid_chunk, '(A,I0)') 'n_manifest_nodes = ', size(nodes_all)
      write(uid_chunk, '(A,I0)') 'n_propagate_nodes = ', n_nodes
      write(uid_chunk, '(A,A)') 'propagate_ids = ', trim(bsv_propagate_ids)
      close(uid_chunk)
      open(newunit=uid_chunk, file='chunk_weighted_spectrum.dat', status='replace', action='write')
      write(uid_chunk, '(A)') '# partial weighted spectrum sum for this chunk (merge by addition)'
      write(uid_chunk, '(A,I0)') '# n_manifest_nodes = ', size(nodes_all)
      write(uid_chunk, '(A,I0)') '# n_propagate_nodes = ', n_nodes
      write(uid_chunk, '(A,ES25.17)') '# spectrum_scale = ', spectrum_scale
      write(uid_chunk, '(A)') '# iw  harmonic_order  omega(a.u.)  weighted_ics  ReSumJx ImSumJx ReSumJy ImSumJy'
      domega = TWOPI / (real(nt, dp) * dt)
      do iw_h = 1, n_omega_s
        omega_n = real(iw_h - 1, dp) * domega
        h_order = omega_n / omega0
        write(uid_chunk,'(I8,ES25.17,ES25.17,ES25.17,4ES25.17)') iw_h, h_order, omega_n, &
          ics_tot(iw_h), &
          real(sum_Jw_x(iw_h), dp), aimag(sum_Jw_x(iw_h)), &
          real(sum_Jw_y(iw_h), dp), aimag(sum_Jw_y(iw_h))
      end do
      close(uid_chunk)
      write(*,'(A)') 'Wrote chunk_info.txt + chunk_weighted_spectrum.dat (partial chunk; merge required).'
    end if
    write(*,'(A)') 'Wrote HHG_nodes_modes.dat (target-harmonic complex amplitudes).'

    deallocate(Jt_sample, sum_Jw_x, sum_Jw_y, ics_tot, cs_tot, var_tot, nodes, nodes_all, harm_orders)
    if (allocated(prop_ids)) deallocate(prop_ids)
    write(*,'(A)') 'Quantum-light ensemble complete.'
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

  subroutine write_bsv_hhg(filename, hhg_avg, hhg_stderr, nw, n_samp, nt_in, dt_in, omega0_in, &
                           state_label_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: hhg_avg(:), hhg_stderr(:)
    integer,      intent(in) :: nw, n_samp, nt_in
    real(dp),     intent(in) :: dt_in, omega0_in
    character(*), intent(in) :: state_label_in
    integer  :: u, iw
    real(dp) :: domega, omega_n, h_order
    domega = TWOPI / (real(nt_in, dp) * dt_in)
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A,I0)') '# n_nodes = ', n_samp
    write(u, '(A,A)')  '# state_type = ', trim(state_label_in)
    write(u, '(A)') '# NOTE: HHG_bsv.dat is the weighted ICS average (layer A).'
    write(u, '(A)') '# harmonic_order  omega(a.u.)  HHG_ics_avg  HHG_ics_stderr'
    do iw = 1, nw
      omega_n = real(iw - 1, dp) * domega
      h_order = omega_n / omega0_in
      write(u, '(F10.4, 3ES16.8)') h_order, omega_n, hhg_avg(iw), hhg_stderr(iw)
    end do
    close(u)
  end subroutine write_bsv_hhg

end program hhg_sbe_solver
