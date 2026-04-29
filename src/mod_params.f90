module mod_params
  implicit none

  integer, parameter :: dp = selected_real_kind(15, 307)
  complex(dp), parameter :: C_I = (0.0_dp, 1.0_dp)
  complex(dp), parameter :: C_0 = (0.0_dp, 0.0_dp)
  complex(dp), parameter :: C_1 = (1.0_dp, 0.0_dp)

  real(dp), parameter :: PI      = 3.14159265358979323846264338327950288_dp
  real(dp), parameter :: TWOPI   = 2.0_dp * PI

  ! --- Atomic-unit conversion constants ---
  real(dp), parameter :: eV_to_Ha    = 1.0_dp / 27.211386245988_dp
  real(dp), parameter :: Ha_to_eV    = 27.211386245988_dp
  real(dp), parameter :: Ang_to_bohr = 1.0_dp / 0.529177210903_dp
  real(dp), parameter :: bohr_to_Ang = 0.529177210903_dp
  real(dp), parameter :: fs_to_au    = 1.0_dp / 0.024188843265857_dp
  real(dp), parameter :: au_to_fs    = 0.024188843265857_dp
  real(dp), parameter :: nm_to_bohr  = 10.0_dp * Ang_to_bohr
  real(dp), parameter :: c_au        = 137.035999084_dp
  real(dp), parameter :: Wcm2_to_au  = 1.0_dp / 3.50944758e16_dp
  real(dp), parameter :: E_au_to_SI  = 5.14220674763e11_dp

  ! --- Crystal / Wannier ---
  real(dp) :: a1_ang(3) = 0.0_dp, a2_ang(3) = 0.0_dp, a3_ang(3) = 0.0_dp
  real(dp) :: a1(3), a2(3), a3(3)
  real(dp) :: b1(3), b2(3), b3(3)
  real(dp) :: Omega_cell
  character(256) :: wannier_tb_file = ''
  character(256) :: wannier_hr_file = ''
  character(256) :: wannier_r_file  = ''
  real(dp) :: E_fermi_eV = 0.0_dp
  real(dp) :: E_fermi    = 0.0_dp
  integer  :: SOC = 1

  ! --- Band truncation ---
  integer :: nv_orig  = 0
  integer :: nb_start = -1
  integer :: nb_end   = -1
  integer :: nwann    = 0
  integer :: n_trunc  = 0
  integer :: nv       = 0

  ! --- k-grid ---
  integer :: nkx = 30, nky = 30

  ! --- Laser ---
  real(dp) :: wvl_nm         = 3200.0_dp
  real(dp) :: intensity_Wcm2 = 2.0e11_dp
  real(dp) :: theta_deg      = 0.0_dp
  real(dp) :: phi_cep_deg    = 90.0_dp
  real(dp) :: ncyc           = 10.0_dp
  integer  :: env_type       = 2
  real(dp) :: ellipticity    = 0.0_dp
  real(dp) :: delta_phase_deg = 90.0_dp
  real(dp) :: omega0, E0, A0, T_cycle, T_total, T_total_1
  real(dp) :: theta, phi_cep, delta_phase
  real(dp) :: pol_vec(3)

  ! --- Laser 2 (dual-color, disabled when wvl_nm_2 = 0) ---
  real(dp) :: wvl_nm_2          = 0.0_dp
  real(dp) :: intensity_Wcm2_2  = 0.0_dp
  real(dp) :: theta_deg_2       = 0.0_dp
  real(dp) :: phi_cep_deg_2     = 0.0_dp
  real(dp) :: ncyc_2            = 0.0_dp
  integer  :: env_type_2        = 2
  real(dp) :: ellipticity_2     = 0.0_dp
  real(dp) :: delta_phase_deg_2 = 90.0_dp
  real(dp) :: omega0_2, E0_2, T_cycle_2, T_total_2
  real(dp) :: theta_2, phi_cep_2, delta_phase_2
  real(dp) :: pol_vec_2(3)
  logical  :: dual_color = .false.

  ! --- Time ---
  real(dp) :: dt         = 0.35_dp
  integer  :: nt         = 0
  integer  :: n_dt_deph  = 5

  ! --- Dephasing ---
  real(dp) :: T2_fs = 10.0_dp
  real(dp) :: T2    = 0.0_dp

  ! --- BSV ---
  logical  :: bsv_enabled        = .false.
  integer  :: bsv_n_samples      = 100
  real(dp) :: bsv_mean_intensity = 0.0_dp
  integer  :: bsv_seed           = 42

  ! --- Method ---
  character(8) :: gauge_method = 'vg'

  ! --- Namelists ---
  namelist /crystal/   a1_ang, a2_ang, a3_ang, E_fermi_eV, SOC, &
                       wannier_tb_file, wannier_hr_file, wannier_r_file
  namelist /kgrid/     nkx, nky
  namelist /bands/     nv_orig, nb_start, nb_end
  namelist /laser/     wvl_nm, intensity_Wcm2, theta_deg, phi_cep_deg, &
                       ncyc, env_type, ellipticity, delta_phase_deg
  namelist /laser2/    wvl_nm_2, intensity_Wcm2_2, theta_deg_2, phi_cep_deg_2, &
                       ncyc_2, env_type_2, ellipticity_2, delta_phase_deg_2
  namelist /timestep/  dt, n_dt_deph
  namelist /dephasing/ T2_fs
  namelist /bsv/       bsv_enabled, bsv_n_samples, bsv_mean_intensity, bsv_seed
  namelist /method/    gauge_method

contains

  subroutine read_input(filename)
    character(*), intent(in) :: filename
    integer :: u, ios
    real(dp) :: v(3)

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: cannot open input file: ', trim(filename)
      error stop 1
    end if

    read(u, nml=crystal,   iostat=ios); rewind(u)
    read(u, nml=kgrid,     iostat=ios); rewind(u)
    read(u, nml=bands,     iostat=ios); rewind(u)
    read(u, nml=laser,     iostat=ios); rewind(u)
    read(u, nml=laser2,    iostat=ios); rewind(u)
    read(u, nml=timestep,  iostat=ios); rewind(u)
    read(u, nml=dephasing, iostat=ios); rewind(u)
    read(u, nml=bsv,       iostat=ios); rewind(u)
    read(u, nml=method,    iostat=ios)
    close(u)

    a1 = a1_ang * Ang_to_bohr
    a2 = a2_ang * Ang_to_bohr
    a3 = a3_ang * Ang_to_bohr

    call cross3(a2, a3, v)
    Omega_cell = abs(dot_product(a1, v))
    b1 = TWOPI * v / dot_product(a1, v)
    call cross3(a3, a1, v)
    b2 = TWOPI * v / dot_product(a2, v)
    call cross3(a1, a2, v)
    b3 = TWOPI * v / dot_product(a3, v)

    E_fermi = E_fermi_eV * eV_to_Ha

    omega0    = TWOPI * c_au / (wvl_nm * nm_to_bohr)
    E0        = sqrt(intensity_Wcm2 * Wcm2_to_au)
    A0        = E0 / omega0
    T_cycle   = TWOPI / omega0
    T_total_1 = ncyc * T_cycle
    T_total   = T_total_1

    theta       = theta_deg       * PI / 180.0_dp
    phi_cep     = phi_cep_deg     * PI / 180.0_dp
    delta_phase = delta_phase_deg * PI / 180.0_dp
    pol_vec = [cos(theta), sin(theta), 0.0_dp]

    dual_color = (wvl_nm_2 > 0.0_dp .and. intensity_Wcm2_2 > 0.0_dp)
    if (dual_color) then
      omega0_2  = TWOPI * c_au / (wvl_nm_2 * nm_to_bohr)
      E0_2      = sqrt(intensity_Wcm2_2 * Wcm2_to_au)
      T_cycle_2 = TWOPI / omega0_2
      if (ncyc_2 <= 0.0_dp) ncyc_2 = ncyc
      T_total_2 = ncyc_2 * T_cycle_2
      T_total   = max(T_total_1, T_total_2)
      theta_2       = theta_deg_2       * PI / 180.0_dp
      phi_cep_2     = phi_cep_deg_2     * PI / 180.0_dp
      delta_phase_2 = delta_phase_deg_2 * PI / 180.0_dp
      pol_vec_2 = [cos(theta_2), sin(theta_2), 0.0_dp]
    end if

    nt = nint(T_total / dt)
    if (nt < 1) nt = 1

    T2 = T2_fs * fs_to_au
  end subroutine read_input

  subroutine finalize_band_params()
    if (nb_start < 1) nb_start = 1
    if (nb_end < 1 .or. nb_end > nwann) nb_end = nwann
    n_trunc = nb_end - nb_start + 1
    nv = nv_orig - nb_start + 1
    if (nv < 1) then
      write(*,*) 'ERROR: nv < 1 after truncation. Check nb_start and nv_orig.'
      error stop 1
    end if
    if (nv > n_trunc) then
      write(*,*) 'ERROR: nv > n_trunc. Check nb_end and nv_orig.'
      error stop 1
    end if
  end subroutine finalize_band_params

  subroutine print_params()
    write(*,'(A)')       '==========================================='
    write(*,'(A)')       '  HHG-SBE Solver Parameters'
    write(*,'(A)')       '==========================================='
    write(*,'(A,3F12.6)')'  a1 (bohr)      : ', a1
    write(*,'(A,3F12.6)')'  a2 (bohr)      : ', a2
    write(*,'(A,3F12.6)')'  b1 (1/bohr)    : ', b1
    write(*,'(A,3F12.6)')'  b2 (1/bohr)    : ', b2
    write(*,'(A,F12.6)') '  Omega_cell     : ', Omega_cell
    write(*,'(A,F12.6)') '  E_fermi (Ha)   : ', E_fermi
    write(*,'(A,I0)')    '  SOC            : ', SOC
    write(*,'(A)')       '-------------------------------------------'
    write(*,'(A,I0)')    '  nwann          : ', nwann
    write(*,'(A,I0,A,I0)') '  Band window : ', nb_start, ' to ', nb_end
    write(*,'(A,I0)')    '  n_trunc        : ', n_trunc
    write(*,'(A,I0)')    '  nv             : ', nv
    write(*,'(A)')       '-------------------------------------------'
    write(*,'(A,I0,A,I0)') '  k-grid       : ', nkx, ' x ', nky
    write(*,'(A)')       '-------------------------------------------'
    write(*,'(A,F10.2,A)') '  Wavelength   : ', wvl_nm, ' nm'
    write(*,'(A,ES10.3,A)')'  Intensity    : ', intensity_Wcm2, ' W/cm^2'
    write(*,'(A,F12.8,A)') '  omega0       : ', omega0, ' a.u.'
    write(*,'(A,F12.8,A)') '  E0           : ', E0, ' a.u.'
    write(*,'(A,F12.8,A)') '  A0           : ', A0, ' a.u.'
    write(*,'(A,F10.2,A)') '  T_cycle      : ', T_cycle * au_to_fs, ' fs'
    write(*,'(A,F10.2,A)') '  T_total      : ', T_total * au_to_fs, ' fs'
    write(*,'(A,I0)')      '  nt           : ', nt
    write(*,'(A,F10.4,A)') '  dt           : ', dt, ' a.u.'
    if (abs(ellipticity) > 1.0e-10_dp) then
      write(*,'(A,F10.4)')   '  ellipticity  : ', ellipticity
      write(*,'(A,F10.2,A)') '  delta_phase  : ', delta_phase_deg, ' deg'
    end if
    if (dual_color) then
      write(*,'(A)')       '--- Laser 2 (dual-color) ------------------'
      write(*,'(A,F10.2,A)') '  Wavelength_2 : ', wvl_nm_2, ' nm'
      write(*,'(A,ES10.3,A)')'  Intensity_2  : ', intensity_Wcm2_2, ' W/cm^2'
      write(*,'(A,F12.8,A)') '  omega0_2     : ', omega0_2, ' a.u.'
      write(*,'(A,F12.8,A)') '  E0_2         : ', E0_2, ' a.u.'
      write(*,'(A,F10.2,A)') '  theta_2      : ', theta_deg_2, ' deg'
      write(*,'(A,F10.2,A)') '  phi_cep_2    : ', phi_cep_deg_2, ' deg'
      if (abs(ellipticity_2) > 1.0e-10_dp) then
        write(*,'(A,F10.4)')   '  ellipticity_2: ', ellipticity_2
      end if
    end if
    write(*,'(A)')       '-------------------------------------------'
    write(*,'(A,A)')       '  gauge        : ', trim(gauge_method)
    write(*,'(A,F10.2,A)') '  T2           : ', T2_fs, ' fs'
    write(*,'(A,I0)')      '  n_dt_deph    : ', n_dt_deph
    write(*,'(A)')       '==========================================='
  end subroutine print_params

  subroutine cross3(a, b, c)
    real(dp), intent(in)  :: a(3), b(3)
    real(dp), intent(out) :: c(3)
    c(1) = a(2)*b(3) - a(3)*b(2)
    c(2) = a(3)*b(1) - a(1)*b(3)
    c(3) = a(1)*b(2) - a(2)*b(1)
  end subroutine cross3

end module mod_params
