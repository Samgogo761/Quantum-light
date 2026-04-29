module mod_crystal
  use mod_params
  use mod_wannier
  implicit none

  real(dp),    allocatable :: kpts_cart(:,:,:)    ! (3, nkx, nky)
  real(dp),    allocatable :: Ek(:,:,:)           ! (n_trunc, nkx, nky)
  complex(dp), allocatable :: U_trunc(:,:,:,:)    ! (nwann, n_trunc, nkx, nky)
  integer,     allocatable :: valley_id(:,:)      ! (nkx, nky) 1=K, 2=K'

contains

  subroutine setup_kgrid()
    integer :: ikx, iky
    real(dp) :: f1, f2

    allocate(kpts_cart(3, nkx, nky))
    do iky = 1, nky
      do ikx = 1, nkx
        f1 = real(ikx - 1, dp) / real(nkx, dp)
        f2 = real(iky - 1, dp) / real(nky, dp)
        kpts_cart(:, ikx, iky) = f1 * b1 + f2 * b2
      end do
    end do
  end subroutine setup_kgrid

  subroutine compute_band_structure()
    integer :: ikx, iky, info
    complex(dp), allocatable :: Hk(:,:), eigvecs(:,:)
    real(dp),    allocatable :: eigvals(:)
    complex(dp), allocatable :: work(:)
    real(dp),    allocatable :: rwork(:)
    integer,     allocatable :: iwork(:)
    integer :: lwork, lrwork, liwork

    call finalize_band_params()

    allocate(Ek(n_trunc, nkx, nky))
    allocate(U_trunc(nwann, n_trunc, nkx, nky))

    lwork  = 2 * nwann + nwann * nwann
    lrwork = 1 + 5 * nwann + 2 * nwann * nwann
    liwork = 3 + 5 * nwann

    !$OMP PARALLEL PRIVATE(ikx, iky, Hk, eigvals, eigvecs, work, rwork, iwork, info)
    allocate(Hk(nwann, nwann), eigvals(nwann), eigvecs(nwann, nwann))
    allocate(work(lwork), rwork(lrwork), iwork(liwork))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        call fourier_ham(kpts_cart(:, ikx, iky), Hk)
        eigvecs = Hk
        call zheevd('V', 'U', nwann, eigvecs, nwann, eigvals, &
                    work, lwork, rwork, lrwork, iwork, liwork, info)
        if (info /= 0) then
          write(*,*) 'ERROR: zheevd failed at k-point', ikx, iky, ' info=', info
          error stop 1
        end if

        Ek(:, ikx, iky) = eigvals(nb_start:nb_end)
        U_trunc(:, :, ikx, iky) = eigvecs(:, nb_start:nb_end)
      end do
    end do
    !$OMP END DO

    deallocate(Hk, eigvals, eigvecs, work, rwork, iwork)
    !$OMP END PARALLEL

    write(*,'(A,F10.4,A)') '  Band gap at Gamma ~', &
      (Ek(nv+1,1,1) - Ek(nv,1,1)) * Ha_to_eV, ' eV'
    write(*,'(A,F10.4,A,F10.4,A)') '  Truncated window: ', &
      Ek(1,1,1)*Ha_to_eV, ' to ', Ek(n_trunc,1,1)*Ha_to_eV, ' eV (at Gamma)'
  end subroutine compute_band_structure

  subroutine project_to_trunc(Mk_full, ikx, iky, Mk_trunc)
    complex(dp), intent(in)  :: Mk_full(nwann, nwann)
    integer,     intent(in)  :: ikx, iky
    complex(dp), intent(out) :: Mk_trunc(n_trunc, n_trunc)
    complex(dp) :: tmp(nwann, n_trunc)

    call zgemm('N', 'N', nwann, n_trunc, nwann, C_1, &
               Mk_full, nwann, U_trunc(:,:,ikx,iky), nwann, C_0, tmp, nwann)
    call zgemm('C', 'N', n_trunc, n_trunc, nwann, C_1, &
               U_trunc(:,:,ikx,iky), nwann, tmp, nwann, C_0, Mk_trunc, n_trunc)
  end subroutine project_to_trunc

  subroutine project_vec_to_trunc(Vk_full, ikx, iky, Vk_trunc)
    complex(dp), intent(in)  :: Vk_full(nwann, nwann, 3)
    integer,     intent(in)  :: ikx, iky
    complex(dp), intent(out) :: Vk_trunc(n_trunc, n_trunc, 3)
    integer :: a
    do a = 1, 3
      call project_to_trunc(Vk_full(:,:,a), ikx, iky, Vk_trunc(:,:,a))
    end do
  end subroutine project_vec_to_trunc

  subroutine project_to_trunc_withU(Mk_full, Uk, nw, ns, Mk_trunc)
    integer,     intent(in)  :: nw, ns
    complex(dp), intent(in)  :: Mk_full(nw, nw)
    complex(dp), intent(in)  :: Uk(nw, ns)
    complex(dp), intent(out) :: Mk_trunc(ns, ns)
    complex(dp) :: tmp(nw, ns)

    call zgemm('N', 'N', nw, ns, nw, C_1, Mk_full, nw, Uk, nw, C_0, tmp, nw)
    call zgemm('C', 'N', ns, ns, nw, C_1, Uk, nw, tmp, nw, C_0, Mk_trunc, ns)
  end subroutine project_to_trunc_withU

  subroutine write_bands(filename)
    character(*), intent(in) :: filename
    integer :: u, ikx, iky, n
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A,I0,A,I0,A,I0)') '# nkx=', nkx, ' nky=', nky, ' n_trunc=', n_trunc
    write(u, '(A)') '# ikx iky n  E(eV)  kx(1/bohr)  ky(1/bohr)'
    do iky = 1, nky
      do ikx = 1, nkx
        do n = 1, n_trunc
          write(u, '(3I6, 3ES16.8)') ikx, iky, n, &
            Ek(n, ikx, iky) * Ha_to_eV, &
            kpts_cart(1, ikx, iky), kpts_cart(2, ikx, iky)
        end do
      end do
    end do
    close(u)
  end subroutine write_bands

  subroutine compute_valley_assignment()
    integer :: ikx, iky, n1, n2
    real(dp) :: K_cart(3), Kp_cart(3), dv(3)
    real(dp) :: min_dist_K, min_dist_Kp, dist

    K_cart  = (2.0_dp/3.0_dp) * b1 + (1.0_dp/3.0_dp) * b2
    Kp_cart = (1.0_dp/3.0_dp) * b1 + (2.0_dp/3.0_dp) * b2

    allocate(valley_id(nkx, nky))

    do iky = 1, nky
      do ikx = 1, nkx
        min_dist_K  = huge(1.0_dp)
        min_dist_Kp = huge(1.0_dp)
        do n2 = -1, 1
          do n1 = -1, 1
            dv = kpts_cart(:,ikx,iky) - K_cart - real(n1,dp)*b1 - real(n2,dp)*b2
            dist = sqrt(dot_product(dv, dv))
            if (dist < min_dist_K) min_dist_K = dist

            dv = kpts_cart(:,ikx,iky) - Kp_cart - real(n1,dp)*b1 - real(n2,dp)*b2
            dist = sqrt(dot_product(dv, dv))
            if (dist < min_dist_Kp) min_dist_Kp = dist
          end do
        end do

        if (min_dist_K <= min_dist_Kp) then
          valley_id(ikx, iky) = 1
        else
          valley_id(ikx, iky) = 2
        end if
      end do
    end do

    write(*,'(A,I0,A,I0)') '  Valley assignment: K=', &
      count(valley_id==1), ', K''=', count(valley_id==2)
  end subroutine compute_valley_assignment

end module mod_crystal
