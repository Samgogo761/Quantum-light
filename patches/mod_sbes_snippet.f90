!===============================================================================
! PATCH SNIPPET for src/mod_sbes.f90
!-------------------------------------------------------------------------------
! Goal:
!   Expose a pure, IO-free "single trajectory" entry point that the MC
!   ensemble driver can call thousands of times without rewriting HHG/
!   output files or colliding with the classical run's artefacts.
!
! Two public additions:
!   init_density_matrix()                -- reset rho -> |v><v| (ground state)
!   sbe_run_single_trajectory(Jt, Jw)    -- RK4 + FFT, no file IO,
!                                           returns J(t) (real) and J(omega)
!                                           (complex) for one E(t) configuration.
!
! Internally re-use your existing sbe_2d_propagation_ht core; all this does is:
!   1. Skip the "open HHG/... for writing" blocks inside the core routine by
!      gating them behind a module-level logical `single_traj_mode`.
!   2. Return J(t) and FFT(J(t)) through the arguments instead of writing
!      them to Jt/ and HHG/.
!===============================================================================

  !-- module-level gate, default .false. so classical path is untouched --------
  logical, save :: single_traj_mode = .false.

  !-----------------------------------------------------------------------------
  ! init_density_matrix
  !   rho_H(:,:,:,:) = 0   except rho_H(v,v,kx,ky) = 1 for valence band v.
  !   Called by mod_ensemble before every trajectory so that each MC sample
  !   starts from the pristine ground state.
  !-----------------------------------------------------------------------------
  subroutine init_density_matrix()
    integer :: ikx, iky, iv
    rho_H = (0.0_dp, 0.0_dp)
    do iky = 1, nky
      do ikx = 1, nkx
        do iv = 1, n_valence
          rho_H(iv, iv, ikx, iky) = (1.0_dp, 0.0_dp)
        end do
      end do
    end do
  end subroutine init_density_matrix

  !-----------------------------------------------------------------------------
  ! sbe_run_single_trajectory
  !   Drop-in replacement for (calc_sbes -> sbe_2d_propagation_ht -> calc_hhg)
  !   when called from the MC driver. Absolutely no file IO: everything comes
  !   back through the argument list.
  !
  !   Jt(1:n_t)   : real-valued time-domain current (sum over x/y components)
  !   Jw(1:n_w)   : complex-valued spectrum = FFT(window(t) * Jt(t))
  !
  !   Caller (mod_ensemble) is responsible for allocation and disposal.
  !-----------------------------------------------------------------------------
  subroutine sbe_run_single_trajectory(Jt, Jw)
    real(dp),    intent(out) :: Jt(:)     ! size = n_t
    complex(dp), intent(out) :: Jw(:)     ! size = n_w

    logical :: prev_mode

    prev_mode        = single_traj_mode
    single_traj_mode = .true.

    ! --- propagation (your existing RK4 core; it should check
    ! --- single_traj_mode and skip all "open / write HHG/..." calls) ---------
    call sbe_2d_propagation_ht()

    ! Extract total current accumulated inside the core (your `jtot(:)` array,
    ! summed over polarizations). Adapt indices to your storage convention.
    Jt(:) = real( jtot(:), dp )

    ! --- FFT via your existing MKL wrapper -----------------------------------
    call mkl_fft_forward(Jt, Jw)         ! your existing wrapper in submod_fft_mkl

    single_traj_mode = prev_mode
  end subroutine sbe_run_single_trajectory
