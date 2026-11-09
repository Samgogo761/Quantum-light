module mod_hhg
  use mod_params
  implicit none

contains

  !-----------------------------------------------------------------------------
  ! Complex Fourier amplitudes of the Hann-windowed current (positive freqs).
  ! Convention: Jw(iw) = FFT[W(t) J(t)], spectrum_scale = dt^2 multiplies |Jw|^2
  ! to match the historical HHG.dat intensity convention.
  !-----------------------------------------------------------------------------
  subroutine compute_complex_current_spectrum(Jt_in, nt_in, dt_in, omega0_in, &
                                              Jw_x, Jw_y, n_omega, spectrum_scale)
    real(dp), intent(in)  :: Jt_in(:,:)
    integer,  intent(in)  :: nt_in
    real(dp), intent(in)  :: dt_in, omega0_in
    complex(dp), allocatable, intent(out) :: Jw_x(:), Jw_y(:)
    integer,  intent(out) :: n_omega
    real(dp), intent(out) :: spectrum_scale

    complex(dp), allocatable :: Jin(:), Jfull_x(:), Jfull_y(:)
    real(dp) :: w_hann
    integer  :: it
    integer(8) :: plan

    if (dt_in <= 0.0_dp) then
      write(*,*) 'ERROR: compute_complex_current_spectrum requires dt_in > 0.'
      error stop 1
    end if
    if (omega0_in <= 0.0_dp) then
      write(*,*) 'ERROR: compute_complex_current_spectrum requires omega0_in > 0.'
      error stop 1
    end if

    n_omega = nt_in / 2 + 1
    spectrum_scale = dt_in * dt_in

    allocate(Jin(nt_in), Jfull_x(nt_in), Jfull_y(nt_in))
    allocate(Jw_x(n_omega), Jw_y(n_omega))

    do it = 1, nt_in
      w_hann = 0.5_dp * (1.0_dp - cos(TWOPI * real(it-1, dp) / real(nt_in-1, dp)))
      Jin(it) = cmplx(Jt_in(it, 1) * w_hann, 0.0_dp, dp)
    end do
    call dfftw_plan_dft_1d(plan, nt_in, Jin, Jfull_x, -1, 64)
    call dfftw_execute_dft(plan, Jin, Jfull_x)
    call dfftw_destroy_plan(plan)

    do it = 1, nt_in
      w_hann = 0.5_dp * (1.0_dp - cos(TWOPI * real(it-1, dp) / real(nt_in-1, dp)))
      Jin(it) = cmplx(Jt_in(it, 2) * w_hann, 0.0_dp, dp)
    end do
    call dfftw_plan_dft_1d(plan, nt_in, Jin, Jfull_y, -1, 64)
    call dfftw_execute_dft(plan, Jin, Jfull_y)
    call dfftw_destroy_plan(plan)

    Jw_x = Jfull_x(1:n_omega)
    Jw_y = Jfull_y(1:n_omega)

    deallocate(Jin, Jfull_x, Jfull_y)
  end subroutine compute_complex_current_spectrum

  subroutine intensities_from_complex(Jw_x, Jw_y, spectrum_scale, hhg_x, hhg_y, hhg_tot)
    complex(dp), intent(in) :: Jw_x(:), Jw_y(:)
    real(dp),    intent(in) :: spectrum_scale
    real(dp), allocatable, intent(out) :: hhg_x(:), hhg_y(:), hhg_tot(:)
    integer :: n_omega, iw

    n_omega = size(Jw_x)
    allocate(hhg_x(n_omega), hhg_y(n_omega), hhg_tot(n_omega))
    do iw = 1, n_omega
      hhg_x(iw)   = spectrum_scale * abs(Jw_x(iw))**2
      hhg_y(iw)   = spectrum_scale * abs(Jw_y(iw))**2
      hhg_tot(iw) = hhg_x(iw) + hhg_y(iw)
    end do
  end subroutine intensities_from_complex

  subroutine compute_hhg_spectrum(Jt_in, nt_in, dt_in, omega0_in, &
                                   hhg_x, hhg_y, hhg_tot, n_omega)
    real(dp), intent(in)  :: Jt_in(:,:)
    integer,  intent(in)  :: nt_in
    real(dp), intent(in)  :: dt_in, omega0_in
    real(dp), allocatable, intent(out) :: hhg_x(:), hhg_y(:), hhg_tot(:)
    integer,  intent(out) :: n_omega

    complex(dp), allocatable :: Jw_x(:), Jw_y(:)
    real(dp) :: spectrum_scale

    call compute_complex_current_spectrum(Jt_in, nt_in, dt_in, omega0_in, &
                                          Jw_x, Jw_y, n_omega, spectrum_scale)
    call intensities_from_complex(Jw_x, Jw_y, spectrum_scale, hhg_x, hhg_y, hhg_tot)
    deallocate(Jw_x, Jw_y)
  end subroutine compute_hhg_spectrum

  subroutine write_hhg(filename, hhg_x, hhg_y, hhg_tot, n_omega, nt_in, dt_in, omega0_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: hhg_x(:), hhg_y(:), hhg_tot(:)
    integer,      intent(in) :: n_omega, nt_in
    real(dp),     intent(in) :: dt_in, omega0_in
    integer  :: u, iw
    real(dp) :: domega, omega_n, h_order

    domega = TWOPI / (real(nt_in, dp) * dt_in)

    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# harmonic_order  omega(a.u.)  HHG_x  HHG_y  HHG_total'
    do iw = 1, n_omega
      omega_n = real(iw - 1, dp) * domega
      h_order = omega_n / omega0_in
      write(u, '(F10.4, 4ES16.8)') h_order, omega_n, hhg_x(iw), hhg_y(iw), hhg_tot(iw)
    end do
    close(u)
  end subroutine write_hhg

  ! Complex amplitudes with the same FFT/window/scale convention as HHG.dat.
  ! Columns include Re/Im of Jw and the intensity |Jw|^2 * spectrum_scale.
  subroutine write_hhg_complex(filename, Jw_x, Jw_y, spectrum_scale, &
                               n_omega, nt_in, dt_in, omega0_in)
    character(*), intent(in) :: filename
    complex(dp),  intent(in) :: Jw_x(:), Jw_y(:)
    real(dp),     intent(in) :: spectrum_scale
    integer,      intent(in) :: n_omega, nt_in
    real(dp),     intent(in) :: dt_in, omega0_in
    integer  :: u, iw
    real(dp) :: domega, omega_n, h_order, sx, sy

    domega = TWOPI / (real(nt_in, dp) * dt_in)
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# Complex harmonic current (Hann-window FFT)'
    write(u, '(A,ES16.8)') '# spectrum_scale = dt^2 = ', spectrum_scale
    write(u, '(A)') '# NOTE: amplitudes are classical trajectory currents, not quantum b_n.'
    write(u, '(A)') '# harmonic_order  omega(a.u.)  ReJx  ImJx  ReJy  ImJy  |Jx|^2_scaled  |Jy|^2_scaled  total'
    do iw = 1, n_omega
      omega_n = real(iw - 1, dp) * domega
      h_order = omega_n / omega0_in
      sx = spectrum_scale * abs(Jw_x(iw))**2
      sy = spectrum_scale * abs(Jw_y(iw))**2
      write(u, '(F10.4, 8ES16.8)') h_order, omega_n, &
        real(Jw_x(iw), dp), aimag(Jw_x(iw)), &
        real(Jw_y(iw), dp), aimag(Jw_y(iw)), &
        sx, sy, sx + sy
    end do
    close(u)
  end subroutine write_hhg_complex

  integer function harmonic_fft_index(nt_in, dt_in, omega0_in, order) result(iw)
    integer, intent(in) :: nt_in, order
    real(dp), intent(in) :: dt_in, omega0_in
    real(dp) :: domega, target_omega
    integer :: n_omega
    n_omega = nt_in / 2 + 1
    domega = TWOPI / (real(nt_in, dp) * dt_in)
    target_omega = real(order, dp) * omega0_in
    iw = nint(target_omega / domega) + 1
    if (iw < 1) iw = 1
    if (iw > n_omega) iw = n_omega
  end function harmonic_fft_index

  subroutine parse_harmonic_list(list, orders, n_ord)
    character(*), intent(in) :: list
    integer, allocatable, intent(out) :: orders(:)
    integer, intent(out) :: n_ord
    character(256) :: buf
    integer :: i, n, v, ios, start
    integer :: tmp(64)
    logical :: in_num
    buf = adjustl(list)
    n = len_trim(buf)
    n_ord = 0
    in_num = .false.
    start = 1
    do i = 1, n + 1
      if (i <= n) then
        if (buf(i:i) >= '0' .and. buf(i:i) <= '9') then
          if (.not. in_num) then
            in_num = .true.
            start = i
          end if
          cycle
        end if
      end if
      if (in_num) then
        read(buf(start:i-1), *, iostat=ios) v
        if (ios == 0 .and. n_ord < 64) then
          n_ord = n_ord + 1
          tmp(n_ord) = v
        end if
        in_num = .false.
      end if
    end do
    if (n_ord <= 0) then
      n_ord = 5
      allocate(orders(5))
      orders = (/ 2, 5, 7, 9, 10 /)
    else
      allocate(orders(n_ord))
      orders = tmp(1:n_ord)
    end if
  end subroutine parse_harmonic_list

  subroutine write_hhg_ics_cs(filename, ics, cs, variance, n_omega, nt_in, dt_in, omega0_in, &
                              n_samp, state_label)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: ics(:), cs(:), variance(:)
    integer,      intent(in) :: n_omega, nt_in, n_samp
    real(dp),     intent(in) :: dt_in, omega0_in
    character(*), intent(in) :: state_label
    integer  :: u, iw
    real(dp) :: domega, omega_n, h_order

    domega = TWOPI / (real(nt_in, dp) * dt_in)
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# Layer-A spectra: ICS / CS / classical_trajectory_variance'
    write(u, '(A)') '# ICS = < |J|^2 >_Q     (incoherent combination of spectra)'
    write(u, '(A)') '# CS  = | <J> |_Q|^2    (coherent sum of complex amplitudes)'
    write(u, '(A)') '# variance = ICS - CS   (NOT quantum source noise; classical benchmark only)'
    write(u, '(A,I0)') '# n_samples = ', n_samp
    write(u, '(A,A)') '# state_type = ', trim(state_label)
    write(u, '(A)') '# harmonic_order  omega(a.u.)  ICS  CS  classical_trajectory_variance'
    do iw = 1, n_omega
      omega_n = real(iw - 1, dp) * domega
      h_order = omega_n / omega0_in
      write(u, '(F10.4, 4ES16.8)') h_order, omega_n, ics(iw), cs(iw), variance(iw)
    end do
    close(u)
  end subroutine write_hhg_ics_cs

end module mod_hhg
