module mod_sbe
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  implicit none

  complex(dp), allocatable :: rho(:,:,:,:)      ! (n_trunc, n_trunc, nkx, nky)
  real(dp),    allocatable :: Jt(:,:)           ! (nt, 2) current output
  real(dp) :: deph_factor

  ! Pre-projected Wannier matrices for O(n_trunc^2) Fourier sums
  complex(dp), allocatable :: HR_proj(:,:,:,:,:)    ! (n_trunc, n_trunc, nrpts, nkx, nky)
  complex(dp), allocatable :: DR_proj(:,:,:,:,:,:)  ! (n_trunc, n_trunc, 3, nrpts, nkx, nky)

contains

  subroutine init_density_matrix()
    integer :: ikx, iky, n

    if (.not. allocated(rho)) then
      allocate(rho(n_trunc, n_trunc, nkx, nky))
    end if
    rho = C_0
    do iky = 1, nky
      do ikx = 1, nkx
        do n = 1, nv
          rho(n, n, ikx, iky) = C_1
        end do
      end do
    end do

    deph_factor = exp(-dt * real(n_dt_deph, dp) / T2)

    if (.not. allocated(Jt)) then
      allocate(Jt(nt, 2))
    end if
    Jt = 0.0_dp
  end subroutine init_density_matrix

  subroutine precompute_projected_matrices()
    integer :: ikx, iky, ir, a, ios
    real(dp) :: kdotR
    complex(dp) :: ph0
    integer(8) :: mem_bytes, mem_MB
    complex(dp), allocatable :: tmp_full(:,:), tmp_proj(:,:)

    mem_bytes = int(n_trunc,8)**2 * int(nrpts,8) * int(nkx,8) * int(nky,8) * 16_8
    if (has_rmn) mem_bytes = mem_bytes * 4_8
    mem_MB = mem_bytes / (1024_8 * 1024_8)
    write(*,'(A,I0,A)') '  Pre-projection memory estimate: ', mem_MB, ' MB'

    allocate(HR_proj(n_trunc, n_trunc, nrpts, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate HR_proj. Out of memory.'
      error stop 1
    end if

    if (has_rmn) then
      allocate(DR_proj(n_trunc, n_trunc, 3, nrpts, nkx, nky), stat=ios)
      if (ios /= 0) then
        write(*,'(A)') '  ERROR: Cannot allocate DR_proj. Out of memory.'
        error stop 1
      end if
    end if

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, ir, a, kdotR, ph0, tmp_full, tmp_proj)
    allocate(tmp_full(nwann, nwann), tmp_proj(n_trunc, n_trunc))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          ph0 = exp(C_I * kdotR) / real(ndegen(ir), dp)

          tmp_full = ph0 * Hmn_R(:,:,ir)
          call project_to_trunc_withU(tmp_full, U_trunc(:,:,ikx,iky), nwann, n_trunc, tmp_proj)
          HR_proj(:,:,ir,ikx,iky) = tmp_proj

          if (has_rmn) then
            do a = 1, 3
              tmp_full = ph0 * rmn_R(:,:,a,ir)
              call project_to_trunc_withU(tmp_full, U_trunc(:,:,ikx,iky), nwann, n_trunc, tmp_proj)
              DR_proj(:,:,a,ir,ikx,iky) = tmp_proj
            end do
          end if
        end do
      end do
    end do
    !$OMP END DO

    deallocate(tmp_full, tmp_proj)
    !$OMP END PARALLEL

    write(*,'(A)') '  Pre-projected Wannier matrices into truncated basis.'
  end subroutine precompute_projected_matrices

  subroutine propagate()
    integer  :: it, ikx, iky, a, m, n, ir
    real(dp) :: Jx_it, Jy_it, Nk_inv
    complex(dp) :: tr_val
    complex(dp), allocatable :: phase_A(:)

    complex(dp), allocatable :: Hk_proj(:,:), Dk_a(:,:)
    complex(dp), allocatable :: vk_a(:,:), Ht(:,:)
    complex(dp), allocatable :: rho_k(:,:), rho_new(:,:)
    complex(dp), allocatable :: k1(:,:), k2(:,:), k3(:,:), k4(:,:)
    complex(dp), allocatable :: rho_tmp(:,:), AB(:,:)

    if (.not. allocated(HR_proj)) then
      write(*,*) 'ERROR: call precompute_projected_matrices() before propagate()'
      error stop 1
    end if

    Nk_inv = 1.0_dp / real(nkx * nky, dp)
    allocate(phase_A(nrpts))

    do it = 1, nt
      Jx_it = 0.0_dp
      Jy_it = 0.0_dp

      call compute_phase_A(At_vec(it,:), phase_A)

      !$OMP PARALLEL DEFAULT(shared) &
      !$OMP PRIVATE(ikx, iky, a, m, n, ir, tr_val, &
      !$OMP         Hk_proj, Dk_a, vk_a, Ht, rho_k, rho_new, &
      !$OMP         k1, k2, k3, k4, rho_tmp, AB) &
      !$OMP REDUCTION(+:Jx_it, Jy_it)

      allocate(Hk_proj(n_trunc,n_trunc), Dk_a(n_trunc,n_trunc))
      allocate(vk_a(n_trunc,n_trunc), Ht(n_trunc,n_trunc))
      allocate(rho_k(n_trunc,n_trunc), rho_new(n_trunc,n_trunc))
      allocate(k1(n_trunc,n_trunc), k2(n_trunc,n_trunc))
      allocate(k3(n_trunc,n_trunc), k4(n_trunc,n_trunc))
      allocate(rho_tmp(n_trunc,n_trunc), AB(n_trunc,n_trunc))

      !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
      do iky = 1, nky
        do ikx = 1, nkx

          ! --- Fourier sum in truncated basis: O(n_trunc^2 * nrpts) ---
          Hk_proj = C_0
          do ir = 1, nrpts
            Hk_proj = Hk_proj + phase_A(ir) * HR_proj(:,:,ir,ikx,iky)
          end do

          Ht = Hk_proj

          if (has_rmn) then
            do a = 1, 3
              if (abs(Et_vec(it, a)) > 1.0e-30_dp) then
                Dk_a = C_0
                do ir = 1, nrpts
                  Dk_a = Dk_a + phase_A(ir) * DR_proj(:,:,a,ir,ikx,iky)
                end do
                Ht = Ht - Et_vec(it, a) * Dk_a
              end if
            end do
          end if

          ! --- RK4 with inlined commutator (no heap alloc per step) ---
          rho_k = rho(:,:,ikx,iky)

          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Ht,n_trunc,rho_k,n_trunc,C_0,AB,n_trunc)
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,rho_k,n_trunc,Ht,n_trunc,C_0,k1,n_trunc)
          k1 = -C_I * (AB - k1)

          rho_tmp = rho_k + 0.5_dp * dt * k1
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Ht,n_trunc,rho_tmp,n_trunc,C_0,AB,n_trunc)
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,rho_tmp,n_trunc,Ht,n_trunc,C_0,k2,n_trunc)
          k2 = -C_I * (AB - k2)

          rho_tmp = rho_k + 0.5_dp * dt * k2
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Ht,n_trunc,rho_tmp,n_trunc,C_0,AB,n_trunc)
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,rho_tmp,n_trunc,Ht,n_trunc,C_0,k3,n_trunc)
          k3 = -C_I * (AB - k3)

          rho_tmp = rho_k + dt * k3
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Ht,n_trunc,rho_tmp,n_trunc,C_0,AB,n_trunc)
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,rho_tmp,n_trunc,Ht,n_trunc,C_0,k4,n_trunc)
          k4 = -C_I * (AB - k4)

          rho_new = rho_k + (dt / 6.0_dp) * (k1 + 2.0_dp*k2 + 2.0_dp*k3 + k4)

          ! --- Dephasing ---
          if (mod(it, n_dt_deph) == 0) then
            do n = 1, n_trunc
              do m = 1, n_trunc
                if (m /= n) rho_new(m,n) = rho_new(m,n) * deph_factor
              end do
            end do
          end if

          rho(:,:,ikx,iky) = rho_new

          ! --- Current: J_a = -Tr(rho * v_a) / Nk ---
          ! v_a(k(t)) = sum_R i*R_a * phase_A(R) * HR_proj(R,k)
          do a = 1, 2
            vk_a = C_0
            do ir = 1, nrpts
              vk_a = vk_a + (C_I * Rvec_cart(a,ir) * phase_A(ir)) * HR_proj(:,:,ir,ikx,iky)
            end do
            tr_val = C_0
            do n = 1, n_trunc
              do m = 1, n_trunc
                tr_val = tr_val + vk_a(m,n) * rho_new(n,m)
              end do
            end do
            if (a == 1) Jx_it = Jx_it + real(tr_val, dp)
            if (a == 2) Jy_it = Jy_it + real(tr_val, dp)
          end do

        end do
      end do
      !$OMP END DO

      deallocate(Hk_proj, Dk_a, vk_a, Ht, rho_k, rho_new)
      deallocate(k1, k2, k3, k4, rho_tmp, AB)
      !$OMP END PARALLEL

      Jt(it, 1) = -Jx_it * Nk_inv
      Jt(it, 2) = -Jy_it * Nk_inv

      if (mod(it, max(nt/20, 1)) == 0) then
        write(*,'(A,I0,A,I0,A,F6.1,A)') '  Step ', it, ' / ', nt, &
          ' (', 100.0_dp*real(it,dp)/real(nt,dp), '%)'
      end if
    end do

    deallocate(phase_A)
  end subroutine propagate

  subroutine compute_phase_A(At_now, phase_A)
    real(dp),    intent(in)  :: At_now(3)
    complex(dp), intent(out) :: phase_A(nrpts)
    integer :: ir
    do ir = 1, nrpts
      phase_A(ir) = exp(C_I * dot_product(At_now, Rvec_cart(:, ir)))
    end do
  end subroutine compute_phase_A

  subroutine run_single_trajectory(Jt_out)
    real(dp), intent(out) :: Jt_out(:,:)
    call init_density_matrix()
    call propagate()
    Jt_out = Jt
  end subroutine run_single_trajectory

end module mod_sbe
