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
    call generate_field_sample(E0, phi_cep)
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

end module mod_laser
