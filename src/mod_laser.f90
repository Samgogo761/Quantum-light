module mod_laser
  use mod_params
  implicit none

  real(dp), allocatable :: Et_vec(:,:)    ! (nt, 3) Cartesian E-field
  real(dp), allocatable :: At_vec(:,:)    ! (nt, 3) Cartesian vector potential

contains

  subroutine generate_field()
    call generate_field_sample(E0, phi_cep)
  end subroutine generate_field

  subroutine generate_field_sample(E_peak, phi_0)
    real(dp), intent(in) :: E_peak, phi_0
    integer :: it, a
    real(dp) :: t, f_env, t_mid
    real(dp) :: ex_dir(3), ey_dir(3)
    real(dp) :: Ex_t, Ey_t

    if (allocated(Et_vec)) deallocate(Et_vec, At_vec)
    allocate(Et_vec(nt, 3), At_vec(nt, 3))

    t_mid = T_total * 0.5_dp

    ex_dir = pol_vec
    ey_dir = [-pol_vec(2), pol_vec(1), 0.0_dp]

    do it = 1, nt
      t = (it - 1) * dt

      select case (env_type)
      case (1)
        f_env = exp(-2.0_dp * log(2.0_dp) * ((t - t_mid) / (T_total * 0.35_dp))**2)
      case (2)
        if (t >= 0.0_dp .and. t <= T_total) then
          f_env = cos(PI * (t - t_mid) / T_total)**2
        else
          f_env = 0.0_dp
        end if
      case (3)
        if (t >= 0.0_dp .and. t <= T_total) then
          f_env = cos(PI * (t - t_mid) / T_total)**4
        else
          f_env = 0.0_dp
        end if
      case default
        f_env = 1.0_dp
      end select

      Ex_t = E_peak * f_env * sin(omega0 * t + phi_0)
      Ey_t = E_peak * ellipticity * f_env * sin(omega0 * t + phi_0 + delta_phase)

      do a = 1, 3
        Et_vec(it, a) = Ex_t * ex_dir(a) + Ey_t * ey_dir(a)
      end do
    end do

    At_vec(1, :) = 0.0_dp
    do it = 2, nt
      do a = 1, 3
        At_vec(it, a) = At_vec(it-1, a) - 0.5_dp * dt * (Et_vec(it-1, a) + Et_vec(it, a))
      end do
    end do
  end subroutine generate_field_sample

end module mod_laser
