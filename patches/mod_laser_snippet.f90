!===============================================================================
! PATCH SNIPPET for src/mod_laser.f90
!-------------------------------------------------------------------------------
! Goal:
!   Refactor the hard-coded cos^2-envelope field builder so that an external
!   driver (mod_ensemble) can hand it an arbitrary peak-E amplitude and a
!   carrier-phase offset. The existing classical entry point
!   calc_laser2d() becomes a thin wrapper that forwards the classical values,
!   preserving bit-compatible behaviour for the regression test in plan §2.
!
! Drop this into mod_laser.f90 (adapting names to your module: the globals
! E_t(:), A_t(:), time grid t_grid(:), envelope params t0, t_fwhm, omega0,
! polarization unit vectors, etc. are taken from your existing code).
!
! Classical regression guarantee:
!   calc_laser2d() -> calc_laser2d_sample(E0_classical, 0.0_dp)
!                    should reproduce the old E(t) / A(t) arrays to RK4-
!                    error level.
!===============================================================================

  !-----------------------------------------------------------------------------
  ! calc_laser2d_sample
  !   Rebuild E(t) and A(t) for a given absolute peak E-field (SI, V/m) and
  !   a carrier phase offset (rad). The envelope shape, polarization axes,
  !   carrier frequency, etc. remain exactly as in the classical path.
  !
  !   E(t) = E_peak_SI * env(t) * cos( omega0 * (t - t0) + phi_0 ) * e_hat
  !   A(t) = -integral E(t') dt'         (trapezoidal, as before)
  !-----------------------------------------------------------------------------
  subroutine calc_laser2d_sample(E_peak_SI, phi_0)
    real(dp), intent(in) :: E_peak_SI   ! absolute peak E-field, V/m
    real(dp), intent(in) :: phi_0       ! carrier phase offset, rad

    integer  :: it
    real(dp) :: t, env, carrier, Ex, Ey

    ! --- rebuild E(t) --------------------------------------------------------
    do it = 1, n_t
      t = t_grid(it)

      ! Re-use the existing envelope. Example for cos^2 (env = 2 in namelist):
      if (abs(t - t0) <= 0.5_dp * t_fwhm) then
        env = cos( PI * (t - t0) / t_fwhm )**2
      else
        env = 0.0_dp
      end if

      carrier = cos( omega0 * (t - t0) + phi_0 )

      Ex = E_peak_SI * env * carrier * e_hat_x
      Ey = E_peak_SI * env * carrier * e_hat_y

      E_t(it, 1) = Ex
      E_t(it, 2) = Ey
    end do

    ! --- derive A(t) = -integral E dt (trapezoid, reuse your existing helper) -
    call integrate_vector_potential(E_t, A_t)   ! <- your existing routine
  end subroutine calc_laser2d_sample

  !-----------------------------------------------------------------------------
  ! calc_laser2d
  !   Legacy classical wrapper. Forwards the namelist "intensity" -> peak E
  !   with zero carrier offset, giving bit-compat with the pre-BSV behaviour.
  !-----------------------------------------------------------------------------
  subroutine calc_laser2d()
    real(dp) :: E0_classical

    ! W/cm^2 -> W/m^2 -> V/m peak amplitude.
    ! (Matches the SI factor used by mod_ensemble.)
    E0_classical = sqrt( 2.0_dp * intensity * 1.0e4_dp &
                         / (2.99792458e8_dp * 8.8541878128e-12_dp) )
    call calc_laser2d_sample(E0_classical, 0.0_dp)
  end subroutine calc_laser2d
