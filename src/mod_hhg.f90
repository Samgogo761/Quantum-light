module mod_hhg
  use mod_params
  implicit none

contains

  subroutine compute_hhg_spectrum(Jt_in, nt_in, dt_in, omega0_in, &
                                   hhg_x, hhg_y, hhg_tot, n_omega)
    real(dp), intent(in)  :: Jt_in(:,:)
    integer,  intent(in)  :: nt_in
    real(dp), intent(in)  :: dt_in, omega0_in
    real(dp), allocatable, intent(out) :: hhg_x(:), hhg_y(:), hhg_tot(:)
    integer,  intent(out) :: n_omega

    complex(dp), allocatable :: Jw_x(:), Jw_y(:), Jin(:)
    real(dp) :: w_hann
    integer  :: it, iw
    integer(8) :: plan

    n_omega = nt_in / 2 + 1

    allocate(Jw_x(nt_in), Jw_y(nt_in), Jin(nt_in))
    allocate(hhg_x(n_omega), hhg_y(n_omega), hhg_tot(n_omega))

    do it = 1, nt_in
      w_hann = 0.5_dp * (1.0_dp - cos(TWOPI * real(it-1, dp) / real(nt_in-1, dp)))
      Jin(it) = cmplx(Jt_in(it, 1) * w_hann, 0.0_dp, dp)
    end do
    call dfftw_plan_dft_1d(plan, nt_in, Jin, Jw_x, -1, 64)
    call dfftw_execute_dft(plan, Jin, Jw_x)
    call dfftw_destroy_plan(plan)

    do it = 1, nt_in
      w_hann = 0.5_dp * (1.0_dp - cos(TWOPI * real(it-1, dp) / real(nt_in-1, dp)))
      Jin(it) = cmplx(Jt_in(it, 2) * w_hann, 0.0_dp, dp)
    end do
    call dfftw_plan_dft_1d(plan, nt_in, Jin, Jw_y, -1, 64)
    call dfftw_execute_dft(plan, Jin, Jw_y)
    call dfftw_destroy_plan(plan)

    do iw = 1, n_omega
      hhg_x(iw)   = abs(Jw_x(iw))**2
      hhg_y(iw)   = abs(Jw_y(iw))**2
      hhg_tot(iw) = hhg_x(iw) + hhg_y(iw)
    end do

    deallocate(Jin, Jw_x, Jw_y)
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

end module mod_hhg
