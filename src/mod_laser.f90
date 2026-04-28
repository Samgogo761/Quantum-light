module mod_laser
  use mod_params
  implicit none

  real(dp), allocatable :: Et_scalar(:)   ! (nt) E-field scalar amplitude
  real(dp), allocatable :: At_scalar(:)   ! (nt) vector potential scalar
  real(dp), allocatable :: Et_vec(:,:)    ! (nt, 3) Cartesian components
  real(dp), allocatable :: At_vec(:,:)    ! (nt, 3) Cartesian components

contains

  subroutine generate_field()
    call generate_field_sample(E0, phi_cep)
  end subroutine generate_field

  subroutine generate_field_sample(E_peak, phi_0)
    real(dp), intent(in) :: E_peak, phi_0
    integer :: it
    real(dp) :: t, f_env, t_mid
    integer :: a

    if (allocated(Et_scalar)) deallocate(Et_scalar, At_scalar, Et_vec, At_vec)
    allocate(Et_scalar(nt), At_scalar(nt), Et_vec(nt, 3), At_vec(nt, 3))

    t_mid = T_total * 0.5_dp

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

      Et_scalar(it) = E_peak * f_env * sin(omega0 * t + phi_0)
    end do

    At_scalar(1) = 0.0_dp
    do it = 2, nt
      At_scalar(it) = At_scalar(it-1) - 0.5_dp * dt * (Et_scalar(it-1) + Et_scalar(it))
    end do

    do a = 1, 3
      Et_vec(:, a) = Et_scalar(:) * pol_vec(a)
      At_vec(:, a) = At_scalar(:) * pol_vec(a)
    end do
  end subroutine generate_field_sample

end module mod_laser
