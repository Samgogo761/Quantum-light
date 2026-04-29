module mod_sbe
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  implicit none

  complex(dp), allocatable :: rho(:,:,:,:)      ! (n_trunc, n_trunc, nkx, nky)
  real(dp),    allocatable :: Jt(:,:)           ! (nt, 2) current output
  real(dp),    allocatable :: Jt_intra(:,:)     ! (nt, 2) intraband current
  real(dp),    allocatable :: Jt_inter(:,:)     ! (nt, 2) interband current
  real(dp),    allocatable :: Jt_K(:,:)         ! (nt, 2) K-valley current
  real(dp),    allocatable :: Jt_Kp(:,:)        ! (nt, 2) K'-valley current
  real(dp) :: deph_factor

  ! Pre-projected Wannier matrices for O(n_trunc^2) Fourier sums
  complex(dp), allocatable :: HR_proj(:,:,:,:,:)    ! (n_trunc, n_trunc, nrpts, nkx, nky)

  ! Length-gauge pre-computed matrices
  complex(dp), allocatable :: Hk_eq(:,:,:,:)        ! (n_trunc, n_trunc, nkx, nky)
  complex(dp), allocatable :: Dk_eq(:,:,:,:,:)      ! (n_trunc, n_trunc, 3, nkx, nky)
  complex(dp), allocatable :: vk_eq(:,:,:,:,:)      ! (n_trunc, n_trunc, 2, nkx, nky)

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

    if (.not. allocated(Jt_intra)) allocate(Jt_intra(nt, 2))
    if (.not. allocated(Jt_inter)) allocate(Jt_inter(nt, 2))
    if (.not. allocated(Jt_K))     allocate(Jt_K(nt, 2))
    if (.not. allocated(Jt_Kp))    allocate(Jt_Kp(nt, 2))
    Jt_intra = 0.0_dp
    Jt_inter = 0.0_dp
    Jt_K     = 0.0_dp
    Jt_Kp    = 0.0_dp
  end subroutine init_density_matrix

  subroutine precompute_projected_matrices()
    integer :: ikx, iky, ir, ios
    real(dp) :: kdotR
    complex(dp) :: ph0
    integer(8) :: mem_bytes, mem_MB
    complex(dp), allocatable :: tmp_full(:,:), tmp_proj(:,:)

    mem_bytes = int(n_trunc,8)**2 * int(nrpts,8) * int(nkx,8) * int(nky,8) * 16_8
    mem_MB = mem_bytes / (1024_8 * 1024_8)
    write(*,'(A,I0,A)') '  Pre-projection memory estimate: ', mem_MB, ' MB'

    allocate(HR_proj(n_trunc, n_trunc, nrpts, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate HR_proj. Out of memory.'
      error stop 1
    end if

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, ir, kdotR, ph0, tmp_full, tmp_proj)
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
        end do
      end do
    end do
    !$OMP END DO

    deallocate(tmp_full, tmp_proj)
    !$OMP END PARALLEL

    write(*,'(A)') '  Pre-projected Wannier matrices into truncated basis.'
  end subroutine precompute_projected_matrices

  subroutine precompute_lg_matrices()
    integer :: ikx, iky, ir, a, ios
    real(dp) :: kdotR
    complex(dp) :: ph0
    complex(dp), allocatable :: tmp_full(:,:), tmp_proj(:,:)

    allocate(Hk_eq(n_trunc, n_trunc, nkx, nky))
    allocate(vk_eq(n_trunc, n_trunc, 2, nkx, nky))

    !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(ikx, iky, ir) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        Hk_eq(:,:,ikx,iky) = C_0
        vk_eq(:,:,1,ikx,iky) = C_0
        vk_eq(:,:,2,ikx,iky) = C_0
        do ir = 1, nrpts
          Hk_eq(:,:,ikx,iky) = Hk_eq(:,:,ikx,iky) + HR_proj(:,:,ir,ikx,iky)
          vk_eq(:,:,1,ikx,iky) = vk_eq(:,:,1,ikx,iky) &
            + (C_I * Rvec_cart(1,ir)) * HR_proj(:,:,ir,ikx,iky)
          vk_eq(:,:,2,ikx,iky) = vk_eq(:,:,2,ikx,iky) &
            + (C_I * Rvec_cart(2,ir)) * HR_proj(:,:,ir,ikx,iky)
        end do
      end do
    end do
    !$OMP END PARALLEL DO

    if (.not. has_rmn) then
      write(*,*) 'ERROR: Length gauge requires dipole matrix (rmn_R). Use tb.dat or provide _r.dat.'
      error stop 1
    end if

    allocate(Dk_eq(n_trunc, n_trunc, 3, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate Dk_eq.'
      error stop 1
    end if
    Dk_eq = C_0

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, ir, a, kdotR, ph0, tmp_full, tmp_proj)
    allocate(tmp_full(nwann, nwann), tmp_proj(n_trunc, n_trunc))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          ph0 = exp(C_I * kdotR) / real(ndegen(ir), dp)
          do a = 1, 3
            tmp_full = ph0 * rmn_R(:,:,a,ir)
            call project_to_trunc_withU(tmp_full, U_trunc(:,:,ikx,iky), nwann, n_trunc, tmp_proj)
            Dk_eq(:,:,a,ikx,iky) = Dk_eq(:,:,a,ikx,iky) + tmp_proj
          end do
        end do
      end do
    end do
    !$OMP END DO

    deallocate(tmp_full, tmp_proj)
    !$OMP END PARALLEL

    write(*,'(A)') '  Pre-computed LG matrices (H_eq, D_eq, v_eq).'
  end subroutine precompute_lg_matrices

  subroutine propagate(do_decompose)
    logical, intent(in), optional :: do_decompose
    logical :: decompose, do_valley, gauge_vg
    integer  :: it, ikx, iky, a, m, n, ir, info_d
    real(dp) :: Jx_it, Jy_it, Nk_inv
    real(dp) :: Jx_intra_it, Jy_intra_it, Jx_inter_it, Jy_inter_it
    real(dp) :: Jx_K_it, Jy_K_it, Jx_Kp_it, Jy_Kp_it
    complex(dp) :: tr_val, tr_intra, tr_inter
    complex(dp), allocatable :: phase_A(:)
    integer :: lwork_d, lrwork_d, liwork_d

    complex(dp), allocatable :: Ht(:,:), vk_a(:,:)
    complex(dp), allocatable :: rho_k(:,:), rho_new(:,:)
    complex(dp), allocatable :: k1(:,:), k2(:,:), k3(:,:), k4(:,:)
    complex(dp), allocatable :: rho_tmp(:,:), AB(:,:)
    complex(dp), allocatable :: Wmat(:,:), rho_band(:,:), v_band(:,:), tmp_mat(:,:)
    real(dp),    allocatable :: eig_tmp(:)
    complex(dp), allocatable :: work_d(:)
    real(dp),    allocatable :: rwork_d(:)
    integer,     allocatable :: iwork_d(:)

    decompose = .true.
    if (present(do_decompose)) decompose = do_decompose
    do_valley = allocated(valley_id) .and. decompose
    gauge_vg = (trim(gauge_method) == 'vg')

    if (.not. gauge_vg .and. trim(gauge_method) /= 'lg') then
      write(*,*) 'ERROR: gauge_method must be "vg" or "lg", got: ', trim(gauge_method)
      error stop 1
    end if

    if (gauge_vg .and. .not. allocated(HR_proj)) then
      write(*,*) 'ERROR: call precompute_projected_matrices() before propagate()'
      error stop 1
    end if

    if (.not. gauge_vg .and. .not. allocated(Hk_eq)) then
      write(*,*) 'ERROR: call precompute_lg_matrices() before propagate() in LG mode'
      error stop 1
    end if

    Nk_inv = 1.0_dp / real(nkx * nky, dp)
    if (gauge_vg) allocate(phase_A(nrpts))

    lwork_d  = 2 * n_trunc + n_trunc * n_trunc
    lrwork_d = 1 + 5 * n_trunc + 2 * n_trunc * n_trunc
    liwork_d = 3 + 5 * n_trunc

    do it = 1, nt
      Jx_it = 0.0_dp;       Jy_it = 0.0_dp
      Jx_intra_it = 0.0_dp; Jy_intra_it = 0.0_dp
      Jx_inter_it = 0.0_dp; Jy_inter_it = 0.0_dp
      Jx_K_it = 0.0_dp;     Jy_K_it = 0.0_dp
      Jx_Kp_it = 0.0_dp;    Jy_Kp_it = 0.0_dp

      if (gauge_vg) call compute_phase_A(At_vec(it,:), phase_A)

      !$OMP PARALLEL DEFAULT(shared) &
      !$OMP PRIVATE(ikx, iky, a, m, n, ir, tr_val, tr_intra, tr_inter, info_d, &
      !$OMP         Ht, vk_a, rho_k, rho_new, &
      !$OMP         k1, k2, k3, k4, rho_tmp, AB, &
      !$OMP         Wmat, eig_tmp, rho_band, v_band, tmp_mat, &
      !$OMP         work_d, rwork_d, iwork_d) &
      !$OMP REDUCTION(+:Jx_it, Jy_it, &
      !$OMP           Jx_intra_it, Jy_intra_it, Jx_inter_it, Jy_inter_it, &
      !$OMP           Jx_K_it, Jy_K_it, Jx_Kp_it, Jy_Kp_it)

      allocate(Ht(n_trunc,n_trunc), vk_a(n_trunc,n_trunc))
      allocate(rho_k(n_trunc,n_trunc), rho_new(n_trunc,n_trunc))
      allocate(k1(n_trunc,n_trunc), k2(n_trunc,n_trunc))
      allocate(k3(n_trunc,n_trunc), k4(n_trunc,n_trunc))
      allocate(rho_tmp(n_trunc,n_trunc), AB(n_trunc,n_trunc))
      if (decompose) then
        allocate(Wmat(n_trunc,n_trunc), eig_tmp(n_trunc))
        allocate(rho_band(n_trunc,n_trunc), v_band(n_trunc,n_trunc))
        allocate(tmp_mat(n_trunc,n_trunc))
        allocate(work_d(lwork_d), rwork_d(lrwork_d), iwork_d(liwork_d))
      end if

      !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
      do iky = 1, nky
        do ikx = 1, nkx

          ! --- Build Ht (gauge-dependent) ---
          if (gauge_vg) then
            Ht = C_0
            do ir = 1, nrpts
              Ht = Ht + phase_A(ir) * HR_proj(:,:,ir,ikx,iky)
            end do
          else
            Ht = Hk_eq(:,:,ikx,iky)
            do a = 1, 3
              if (abs(Et_vec(it,a)) > 1.0e-30_dp) then
                Ht = Ht - Et_vec(it,a) * Dk_eq(:,:,a,ikx,iky)
              end if
            end do
          end if

          ! --- RK4 with inlined commutator ---
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

          ! --- Diagonalize for intra/inter decomposition ---
          if (decompose) then
            Wmat = Ht
            call zheevd('V', 'U', n_trunc, Wmat, n_trunc, eig_tmp, &
                        work_d, lwork_d, rwork_d, lrwork_d, iwork_d, liwork_d, info_d)
            call zgemm('C','N',n_trunc,n_trunc,n_trunc,C_1, &
                       Wmat,n_trunc,rho_new,n_trunc,C_0,tmp_mat,n_trunc)
            call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1, &
                       tmp_mat,n_trunc,Wmat,n_trunc,C_0,rho_band,n_trunc)
          end if

          ! --- Current (gauge-dependent velocity) ---
          do a = 1, 2
            if (gauge_vg) then
              vk_a = C_0
              do ir = 1, nrpts
                vk_a = vk_a + (C_I * Rvec_cart(a,ir) * phase_A(ir)) * HR_proj(:,:,ir,ikx,iky)
              end do
            else
              vk_a = vk_eq(:,:,a,ikx,iky)
            end if

            tr_val = C_0
            do n = 1, n_trunc
              do m = 1, n_trunc
                tr_val = tr_val + vk_a(m,n) * rho_new(n,m)
              end do
            end do

            if (decompose) then
              call zgemm('C','N',n_trunc,n_trunc,n_trunc,C_1, &
                         Wmat,n_trunc,vk_a,n_trunc,C_0,tmp_mat,n_trunc)
              call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1, &
                         tmp_mat,n_trunc,Wmat,n_trunc,C_0,v_band,n_trunc)

              tr_intra = C_0
              tr_inter = C_0
              do n = 1, n_trunc
                tr_intra = tr_intra + v_band(n,n) * rho_band(n,n)
                do m = 1, n_trunc
                  if (m /= n) tr_inter = tr_inter + v_band(m,n) * rho_band(n,m)
                end do
              end do
            end if

            if (a == 1) then
              Jx_it = Jx_it + real(tr_val, dp)
              if (decompose) then
                Jx_intra_it = Jx_intra_it + real(tr_intra, dp)
                Jx_inter_it = Jx_inter_it + real(tr_inter, dp)
              end if
              if (do_valley) then
                if (valley_id(ikx,iky) == 1) then
                  Jx_K_it = Jx_K_it + real(tr_val, dp)
                else
                  Jx_Kp_it = Jx_Kp_it + real(tr_val, dp)
                end if
              end if
            else
              Jy_it = Jy_it + real(tr_val, dp)
              if (decompose) then
                Jy_intra_it = Jy_intra_it + real(tr_intra, dp)
                Jy_inter_it = Jy_inter_it + real(tr_inter, dp)
              end if
              if (do_valley) then
                if (valley_id(ikx,iky) == 1) then
                  Jy_K_it = Jy_K_it + real(tr_val, dp)
                else
                  Jy_Kp_it = Jy_Kp_it + real(tr_val, dp)
                end if
              end if
            end if
          end do

        end do
      end do
      !$OMP END DO

      deallocate(Ht, vk_a, rho_k, rho_new)
      deallocate(k1, k2, k3, k4, rho_tmp, AB)
      if (decompose) then
        deallocate(Wmat, eig_tmp, rho_band, v_band, tmp_mat)
        deallocate(work_d, rwork_d, iwork_d)
      end if
      !$OMP END PARALLEL

      Jt(it, 1) = -Jx_it * Nk_inv
      Jt(it, 2) = -Jy_it * Nk_inv

      if (decompose) then
        Jt_intra(it, 1) = -Jx_intra_it * Nk_inv
        Jt_intra(it, 2) = -Jy_intra_it * Nk_inv
        Jt_inter(it, 1) = -Jx_inter_it * Nk_inv
        Jt_inter(it, 2) = -Jy_inter_it * Nk_inv
      end if

      if (do_valley) then
        Jt_K(it, 1)  = -Jx_K_it * Nk_inv
        Jt_K(it, 2)  = -Jy_K_it * Nk_inv
        Jt_Kp(it, 1) = -Jx_Kp_it * Nk_inv
        Jt_Kp(it, 2) = -Jy_Kp_it * Nk_inv
      end if

      if (mod(it, max(nt/20, 1)) == 0) then
        write(*,'(A,I0,A,I0,A,F6.1,A)') '  Step ', it, ' / ', nt, &
          ' (', 100.0_dp*real(it,dp)/real(nt,dp), '%)'
      end if
    end do

    if (gauge_vg) deallocate(phase_A)
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
    call propagate(.false.)
    Jt_out = Jt
  end subroutine run_single_trajectory

end module mod_sbe
