module mod_laser
  use mod_params
  implicit none

  real(dp), allocatable :: Et_vec(:,:)    ! (nt, 3) Cartesian E-field
  real(dp), allocatable :: At_vec(:,:)    ! (nt, 3) Cartesian vector potential

contains

  function envelope(t, t_mid, T_pulse, etype) result(f)
    real(dp), intent(in) :: t, t_mid, T_pulse
    integer,  intent(in) :: etype
    real(dp) :: f

    select case (etype)
    case (1)
      f = exp(-2.0_dp * log(2.0_dp) * ((t - t_mid) / (T_pulse * 0.35_dp))**2)
    case (2)
      if (abs(t - t_mid) <= T_pulse * 0.5_dp) then
        f = cos(PI * (t - t_mid) / T_pulse)**2
      else
        f = 0.0_dp
      end if
    case (3)
      if (abs(t - t_mid) <= T_pulse * 0.5_dp) then
        f = cos(PI * (t - t_mid) / T_pulse)**4
      else
        f = 0.0_dp
      end if
    case default
      f = 1.0_dp
    end select
  end function envelope

  subroutine generate_field()
    if (use_external_A) then
      call read_external_A_field(trim(external_A_file))
    else
      call generate_field_sample(E0, phi_cep)
    end if
  end subroutine generate_field

  subroutine generate_field_sample(E_peak, phi_0)
    real(dp), intent(in) :: E_peak, phi_0
    integer :: it, a
    real(dp) :: t, f_env, t_mid
    real(dp) :: ex_dir(3), ey_dir(3)
    real(dp) :: Ex_t, Ey_t
    real(dp) :: A_end(3), frac

    if (allocated(Et_vec)) deallocate(Et_vec, At_vec)
    allocate(Et_vec(nt, 3), At_vec(nt, 3))

    t_mid = T_total * 0.5_dp

    ex_dir = pol_vec
    ey_dir = [-pol_vec(2), pol_vec(1), 0.0_dp]

    ! --- Primary pulse ---
    do it = 1, nt
      t = (it - 1) * dt
      f_env = envelope(t, t_mid, T_total_1, env_type)

      Ex_t = E_peak * f_env * sin(omega0 * t + phi_0)
      Ey_t = E_peak * ellipticity * f_env * sin(omega0 * t + phi_0 + delta_phase)

      do a = 1, 3
        Et_vec(it, a) = Ex_t * ex_dir(a) + Ey_t * ey_dir(a)
      end do
    end do

    ! --- Second pulse (dual-color, not BSV-modulated) ---
    if (dual_color) then
      block
        real(dp) :: ex2(3), ey2(3), f2, Ex2_t, Ey2_t
        ex2 = pol_vec_2
        ey2 = [-pol_vec_2(2), pol_vec_2(1), 0.0_dp]
        do it = 1, nt
          t = (it - 1) * dt
          f2 = envelope(t, t_mid, T_total_2, env_type_2)

          Ex2_t = E0_2 * f2 * sin(omega0_2 * t + phi_cep_2)
          Ey2_t = E0_2 * ellipticity_2 * f2 * sin(omega0_2 * t + phi_cep_2 + delta_phase_2)

          do a = 1, 3
            Et_vec(it, a) = Et_vec(it, a) + Ex2_t * ex2(a) + Ey2_t * ey2(a)
          end do
        end do
      end block
    end if

    ! --- A(t) = -integral E(t') dt' via trapezoidal rule ---
    At_vec(1, :) = 0.0_dp
    do it = 2, nt
      do a = 1, 3
        At_vec(it, a) = At_vec(it-1, a) - 0.5_dp * dt * (Et_vec(it-1, a) + Et_vec(it, a))
      end do
    end do

    ! Remove the small numerical DC area left by finite time sampling so that
    ! the velocity-gauge pulse starts and ends with the same vector potential.
    if (nt > 1) then
      A_end = At_vec(nt, :)
      do it = 1, nt
        frac = real(it - 1, dp) / real(nt - 1, dp)
        At_vec(it, :) = At_vec(it, :) - frac * A_end
      end do
      if (maxval(abs(A_end)) > 1.0e-10_dp) then
        write(*,'(A,3ES12.4)') '  Corrected residual A(T): ', A_end
      end if
    end if
  end subroutine generate_field_sample

  subroutine read_external_A_field(filename)
    character(*), intent(in) :: filename
    integer :: u, ios, nrow, it, idx
    real(dp) :: t_fs, ax, ay, step_au, max_step_err
    real(dp), allocatable :: tvals_fs(:)

    if (allocated(Et_vec)) deallocate(Et_vec, At_vec)

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: cannot open external A(t) file: ', trim(filename)
      error stop 1
    end if

    nrow = 0
    do
      read(u, *, iostat=ios) idx, t_fs, ax, ay
      if (ios /= 0) exit
      nrow = nrow + 1
    end do
    close(u)

    if (nrow < 2) then
      write(*,*) 'ERROR: external A(t) file has fewer than 2 readable rows.'
      error stop 1
    end if

    allocate(Et_vec(nrow, 3), At_vec(nrow, 3), tvals_fs(nrow))
    Et_vec = 0.0_dp
    At_vec = 0.0_dp

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: cannot reopen external A(t) file: ', trim(filename)
      error stop 1
    end if

    do it = 1, nrow
      read(u, *, iostat=ios) idx, tvals_fs(it), At_vec(it, 1), At_vec(it, 2)
      if (ios /= 0) then
        write(*,*) 'ERROR: failed while reading external A(t) row ', it
        error stop 1
      end if
    end do
    close(u)

    dt = (tvals_fs(2) - tvals_fs(1)) * fs_to_au
    if (dt <= 0.0_dp) then
      write(*,*) 'ERROR: external A(t) file has non-positive time step.'
      error stop 1
    end if

    max_step_err = 0.0_dp
    do it = 2, nrow - 1
      step_au = (tvals_fs(it + 1) - tvals_fs(it)) * fs_to_au
      max_step_err = max(max_step_err, abs(step_au - dt))
    end do
    if (max_step_err > max(1.0e-8_dp, 1.0e-8_dp * dt)) then
      write(*,'(A,ES12.4,A)') '  WARNING: external A(t) grid is not exactly uniform; max dt error = ', &
        max_step_err, ' a.u.'
    end if

    nt = nrow
    T_total = (tvals_fs(nrow) - tvals_fs(1)) * fs_to_au
    T_total_1 = T_total
    ncyc = T_total / T_cycle

    ! For VG diagnostics the vector potential is primary.  E(t) is reconstructed
    ! only for code paths or output that still expect an electric field array.
    Et_vec(1, :) = -(At_vec(2, :) - At_vec(1, :)) / dt
    do it = 2, nt - 1
      Et_vec(it, :) = -(At_vec(it + 1, :) - At_vec(it - 1, :)) / (2.0_dp * dt)
    end do
    Et_vec(nt, :) = -(At_vec(nt, :) - At_vec(nt - 1, :)) / dt

    write(*,'(A)')       '--- External A(t) loaded ------------------'
    write(*,'(A,A)')     '  file         : ', trim(filename)
    write(*,'(A,I0)')    '  nt           : ', nt
    write(*,'(A,F10.4,A)') '  dt           : ', dt, ' a.u.'
    write(*,'(A,F10.5,A)') '  dt           : ', dt * au_to_fs, ' fs'
    write(*,'(A,F10.2,A)') '  T_total      : ', T_total * au_to_fs, ' fs'
    write(*,'(A,3ES12.4)') '  A(t=0)       : ', At_vec(1, :)
    write(*,'(A,3ES12.4)') '  A(t=end)     : ', At_vec(nt, :)

    deallocate(tvals_fs)
  end subroutine read_external_A_field

end module mod_laser
