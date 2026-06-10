module mod_geometry
  !---------------------------------------------------------------------------
  ! Band-resolved quantum geometry for the length-gauge (lg_cov) production
  ! path: Berry curvature Omega_n(k) and quantum metric g_n(k).
  !
  ! Both follow from the band-resolved quantum geometric tensor
  !
  !     Q^{ab}_n(k) = sum_{m/=n}  v^a_{nm}(k) v^b_{mn}(k) / ( E_m(k) - E_n(k) )^2
  !
  ! with   Omega_n   = -2 Im Q^{xy}_n        (Berry curvature, antisymmetric)
  !        g^{ab}_n  =      Re Q^{ab}_n       (quantum metric, symmetric).
  !
  ! Here v^a = Pk_eq(:,:,a) is the gauge-covariant velocity already projected
  ! into the band (eigen) basis by precompute_lg_matrices, and E_n = Ek are the
  ! band energies. Because U_trunc diagonalises H(k), Pk_eq(n,m,a) are the band
  ! velocity matrix elements <n,k|v_a|m,k>, so this is exact within the window.
  !
  ! Physics note (CrI3-AFM): the structure is PT symmetric, which forces
  ! Omega_n(k) = 0 pointwise. Computing it and showing |Omega| ~ numerical floor
  ! is therefore the rigorous demonstration that the anomalous (Berry) current
  ! j_anom vanishes, so even harmonics are carried by the interband polarization
  ! current. The quantum metric g (PT-even) survives and is the geometric object
  ! that can underlie the even-harmonic polarization response.
  !---------------------------------------------------------------------------
  use mod_params
  use mod_crystal, only: Ek, kpts_cart, valley_id
  use mod_sbe,     only: Pk_eq
  implicit none

  real(dp), allocatable :: geo_berry(:,:,:)   ! Omega_n(k)        (n_trunc,nkx,nky)
  real(dp), allocatable :: geo_gxx(:,:,:)     ! g^{xx}_n(k)
  real(dp), allocatable :: geo_gyy(:,:,:)     ! g^{yy}_n(k)
  real(dp), allocatable :: geo_gxy(:,:,:)     ! g^{xy}_n(k)

  real(dp), parameter :: GEO_DEGEN_TOL = 1.0e-8_dp  ! skip near-degenerate pairs

contains

  subroutine compute_quantum_geometry()
    integer :: ikx, iky, m, n
    real(dp) :: dE, inv_dE2
    complex(dp) :: vx_nm, vy_nm, qxy
    real(dp) :: gxx_n, gyy_n

    if (.not. allocated(Pk_eq)) then
      write(*,'(A)') '  WARNING: Pk_eq not allocated; skipping quantum geometry.'
      return
    end if

    if (allocated(geo_berry)) deallocate(geo_berry)
    if (allocated(geo_gxx))   deallocate(geo_gxx)
    if (allocated(geo_gyy))   deallocate(geo_gyy)
    if (allocated(geo_gxy))   deallocate(geo_gxy)
    allocate(geo_berry(n_trunc, nkx, nky))
    allocate(geo_gxx(n_trunc, nkx, nky))
    allocate(geo_gyy(n_trunc, nkx, nky))
    allocate(geo_gxy(n_trunc, nkx, nky))
    geo_berry = 0.0_dp
    geo_gxx   = 0.0_dp
    geo_gyy   = 0.0_dp
    geo_gxy   = 0.0_dp

    !$OMP PARALLEL DO DEFAULT(shared) COLLAPSE(2) SCHEDULE(dynamic) &
    !$OMP   PRIVATE(ikx, iky, m, n, dE, inv_dE2, vx_nm, vy_nm, qxy, gxx_n, gyy_n)
    do iky = 1, nky
      do ikx = 1, nkx
        do n = 1, n_trunc
          qxy   = C_0
          gxx_n = 0.0_dp
          gyy_n = 0.0_dp
          do m = 1, n_trunc
            if (m == n) cycle
            dE = Ek(m, ikx, iky) - Ek(n, ikx, iky)
            if (abs(dE) < GEO_DEGEN_TOL) cycle
            inv_dE2 = 1.0_dp / (dE * dE)
            vx_nm = Pk_eq(n, m, 1, ikx, iky)
            vy_nm = Pk_eq(n, m, 2, ikx, iky)
            ! Q^{xy}_n = sum_m v^x_{nm} v^y_{mn} / dE^2 ; v_{mn} = conjg(v_{nm}) (Hermitian)
            qxy   = qxy + vx_nm * conjg(vy_nm) * inv_dE2
            gxx_n = gxx_n + real(vx_nm * conjg(vx_nm), dp) * inv_dE2
            gyy_n = gyy_n + real(vy_nm * conjg(vy_nm), dp) * inv_dE2
          end do
          geo_berry(n, ikx, iky) = -2.0_dp * aimag(qxy)
          geo_gxx(n, ikx, iky)   = gxx_n
          geo_gyy(n, ikx, iky)   = gyy_n
          geo_gxy(n, ikx, iky)   = real(qxy, dp)
        end do
      end do
    end do
    !$OMP END PARALLEL DO

    call report_geometry_summary()
  end subroutine compute_quantum_geometry

  subroutine report_geometry_summary()
    ! Print BZ-summary numbers, in particular the PT check on Berry curvature.
    real(dp) :: max_abs_omega, sum_abs_omega, max_trg, mean_trg
    integer  :: ikx, iky, n, ncount

    max_abs_omega = 0.0_dp
    sum_abs_omega = 0.0_dp
    max_trg       = 0.0_dp
    mean_trg      = 0.0_dp
    ncount        = 0
    do iky = 1, nky
      do ikx = 1, nkx
        do n = 1, n_trunc
          max_abs_omega = max(max_abs_omega, abs(geo_berry(n, ikx, iky)))
          sum_abs_omega = sum_abs_omega + abs(geo_berry(n, ikx, iky))
          max_trg = max(max_trg, geo_gxx(n, ikx, iky) + geo_gyy(n, ikx, iky))
          mean_trg = mean_trg + (geo_gxx(n, ikx, iky) + geo_gyy(n, ikx, iky))
          ncount = ncount + 1
        end do
      end do
    end do
    if (ncount > 0) mean_trg = mean_trg / real(ncount, dp)

    write(*,'(A)')          '  Quantum geometry computed (Omega + quantum metric).'
    write(*,'(A,ES12.4)')   '    max |Omega_n(k)|      (a.u.) : ', max_abs_omega
    write(*,'(A,ES12.4)')   '    mean|Omega_n(k)|      (a.u.) : ', sum_abs_omega / real(max(ncount,1), dp)
    write(*,'(A)')          '    (PT-symmetric AFM => max|Omega| should sit at the numerical floor)'
    write(*,'(A,ES12.4)')   '    max  Tr g_n(k)        (a.u.) : ', max_trg
    write(*,'(A,ES12.4)')   '    mean Tr g_n(k)        (a.u.) : ', mean_trg
  end subroutine report_geometry_summary

  subroutine write_quantum_geometry(filename)
    character(*), intent(in) :: filename
    integer :: u, ikx, iky, n, vid

    if (.not. allocated(geo_berry)) then
      write(*,'(A)') '  WARNING: geometry not computed; nothing written.'
      return
    end if

    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# Band-resolved quantum geometry (length gauge, band basis)'
    write(u, '(A)') '# ikx iky band  kx(1/bohr) ky(1/bohr)  E(eV)  Omega(a.u.)  gxx  gyy  gxy  valley'
    do iky = 1, nky
      do ikx = 1, nkx
        if (allocated(valley_id)) then
          vid = valley_id(ikx, iky)
        else
          vid = 0
        end if
        do n = 1, n_trunc
          write(u, '(3I6, 7ES16.8, I4)') ikx, iky, n, &
            kpts_cart(1, ikx, iky), kpts_cart(2, ikx, iky), &
            Ek(n, ikx, iky) * Ha_to_eV, &
            geo_berry(n, ikx, iky), geo_gxx(n, ikx, iky), &
            geo_gyy(n, ikx, iky), geo_gxy(n, ikx, iky), vid
        end do
      end do
    end do
    close(u)
    write(*,'(A,A)') '  Quantum geometry written to ', trim(filename)
  end subroutine write_quantum_geometry

end module mod_geometry
