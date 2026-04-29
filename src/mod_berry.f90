module mod_berry
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_sbe, only: HR_proj
  implicit none

  real(dp), allocatable :: berry_curv(:,:,:)   ! (n_trunc, nkx, nky)

contains

  subroutine compute_berry_curvature()
    integer :: ikx, iky, ir, m, n
    complex(dp), allocatable :: vx(:,:), vy(:,:)
    real(dp) :: dE

    allocate(berry_curv(n_trunc, nkx, nky))
    berry_curv = 0.0_dp

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, ir, m, n, vx, vy, dE)
    allocate(vx(n_trunc, n_trunc), vy(n_trunc, n_trunc))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        vx = C_0
        vy = C_0
        do ir = 1, nrpts
          vx = vx + (C_I * Rvec_cart(1, ir)) * HR_proj(:,:,ir,ikx,iky)
          vy = vy + (C_I * Rvec_cart(2, ir)) * HR_proj(:,:,ir,ikx,iky)
        end do

        do n = 1, n_trunc
          berry_curv(n, ikx, iky) = 0.0_dp
          do m = 1, n_trunc
            if (m == n) cycle
            dE = Ek(m, ikx, iky) - Ek(n, ikx, iky)
            if (abs(dE) < 1.0e-10_dp) cycle
            berry_curv(n, ikx, iky) = berry_curv(n, ikx, iky) &
              - 2.0_dp * aimag(vx(n,m) * vy(m,n)) / (dE**2)
          end do
        end do
      end do
    end do
    !$OMP END DO

    deallocate(vx, vy)
    !$OMP END PARALLEL

    write(*,'(A)') '  Berry curvature computed (Kubo formula).'
  end subroutine compute_berry_curvature

  subroutine write_berry_curvature(filename)
    character(*), intent(in) :: filename
    integer :: u, ikx, iky, n

    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# ikx iky band  kx(1/bohr) ky(1/bohr)  Omega(a.u.)  valley'
    do iky = 1, nky
      do ikx = 1, nkx
        do n = 1, n_trunc
          write(u, '(3I6, 3ES16.8, I4)') ikx, iky, n, &
            kpts_cart(1, ikx, iky), kpts_cart(2, ikx, iky), &
            berry_curv(n, ikx, iky), valley_id(ikx, iky)
        end do
      end do
    end do
    close(u)
  end subroutine write_berry_curvature

end module mod_berry
