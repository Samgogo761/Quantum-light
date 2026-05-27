module mod_sbe
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  implicit none

  integer, parameter :: MODE_PEIERLS_VG = 1
  integer, parameter :: MODE_MATRIX_VG  = 2
  integer, parameter :: MODE_LG         = 3
  integer, parameter :: MODE_LG_COV     = 4

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
  complex(dp), allocatable :: Pk_eq(:,:,:,:,:)      ! (n_trunc, n_trunc, 3, nkx, nky)

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
    complex(dp), allocatable :: H_full(:,:), D_full(:,:,:)
    complex(dp), allocatable :: dH_full(:,:,:), P_full(:,:,:)
    complex(dp), allocatable :: tmp_proj(:,:)

    if (.not. has_rmn) then
      write(*,*) 'ERROR: Length gauge requires dipole matrix (rmn_R). Use tb.dat or provide _r.dat.'
      error stop 1
    end if

    if (allocated(Hk_eq)) deallocate(Hk_eq)
    if (allocated(Dk_eq)) deallocate(Dk_eq)
    if (allocated(Pk_eq)) deallocate(Pk_eq)
    if (allocated(vk_eq)) deallocate(vk_eq)

    allocate(Hk_eq(n_trunc, n_trunc, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate Hk_eq for LG.'
      error stop 1
    end if
    allocate(Dk_eq(n_trunc, n_trunc, 3, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate Dk_eq.'
      error stop 1
    end if
    allocate(Pk_eq(n_trunc, n_trunc, 3, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate Pk_eq for LG current.'
      error stop 1
    end if
    allocate(vk_eq(n_trunc, n_trunc, 2, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate vk_eq for LG compatibility.'
      error stop 1
    end if
    Hk_eq = C_0
    Dk_eq = C_0
    Pk_eq = C_0
    vk_eq = C_0

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, ir, a, kdotR, ph0, &
    !$OMP                                  H_full, D_full, dH_full, P_full, tmp_proj)
    allocate(H_full(nwann, nwann), D_full(nwann, nwann, 3))
    allocate(dH_full(nwann, nwann, 3), P_full(nwann, nwann, 3))
    allocate(tmp_proj(n_trunc, n_trunc))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        H_full = C_0
        D_full = C_0
        dH_full = C_0

        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          ph0 = exp(C_I * kdotR) / real(ndegen(ir), dp)
          H_full = H_full + ph0 * Hmn_R(:,:,ir)
          do a = 1, 3
            D_full(:,:,a) = D_full(:,:,a) + ph0 * rmn_R(:,:,a,ir)
            dH_full(:,:,a) = dH_full(:,:,a) &
              + (C_I * Rvec_cart(a,ir) * ph0) * Hmn_R(:,:,ir)
          end do
        end do

        H_full = 0.5_dp * (H_full + conjg(transpose(H_full)))
        call project_to_trunc_withU(H_full, U_trunc(:,:,ikx,iky), nwann, n_trunc, tmp_proj)
        Hk_eq(:,:,ikx,iky) = tmp_proj

        do a = 1, 3
          D_full(:,:,a) = 0.5_dp * (D_full(:,:,a) + conjg(transpose(D_full(:,:,a))))
          dH_full(:,:,a) = 0.5_dp * (dH_full(:,:,a) + conjg(transpose(dH_full(:,:,a))))
          P_full(:,:,a) = dH_full(:,:,a) - C_I * &
            (matmul(D_full(:,:,a), H_full) - matmul(H_full, D_full(:,:,a)))
          P_full(:,:,a) = 0.5_dp * (P_full(:,:,a) + conjg(transpose(P_full(:,:,a))))

          call project_to_trunc_withU(D_full(:,:,a), U_trunc(:,:,ikx,iky), &
                                      nwann, n_trunc, tmp_proj)
          Dk_eq(:,:,a,ikx,iky) = tmp_proj

          call project_to_trunc_withU(P_full(:,:,a), U_trunc(:,:,ikx,iky), &
                                      nwann, n_trunc, tmp_proj)
          Pk_eq(:,:,a,ikx,iky) = tmp_proj
          if (a <= 2) vk_eq(:,:,a,ikx,iky) = tmp_proj
        end do
      end do
    end do
    !$OMP END DO

    deallocate(H_full, D_full, dH_full, P_full, tmp_proj)
    !$OMP END PARALLEL

    write(*,'(A)') '  Pre-computed LG matrices (H_eq, D_eq, P_eq current).'
  end subroutine precompute_lg_matrices

  subroutine precompute_matrix_vg_matrices()
    integer :: ikx, iky, ir, a, ios
    real(dp) :: kdotR
    complex(dp) :: ph0
    integer(8) :: mem_bytes, mem_MB
    complex(dp), allocatable :: H_full(:,:), D_full(:,:,:)
    complex(dp), allocatable :: dH_full(:,:,:), P_full(:,:,:)
    complex(dp), allocatable :: tmp_proj(:,:)

    if (.not. has_rmn) then
      write(*,*) 'ERROR: matrix_vg requires dipole matrix rmn_R from tb.dat or _r.dat.'
      error stop 1
    end if

    mem_bytes = int(n_trunc,8)**2 * int(nkx,8) * int(nky,8) * 4_8 * 16_8
    mem_MB = mem_bytes / (1024_8 * 1024_8)
    write(*,'(A,I0,A)') '  Matrix-VG precompute memory estimate: ', mem_MB, ' MB'

    if (allocated(Hk_eq)) deallocate(Hk_eq)
    if (allocated(Pk_eq)) deallocate(Pk_eq)
    allocate(Hk_eq(n_trunc, n_trunc, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate Hk_eq for matrix_vg.'
      error stop 1
    end if
    allocate(Pk_eq(n_trunc, n_trunc, 3, nkx, nky), stat=ios)
    if (ios /= 0) then
      write(*,'(A)') '  ERROR: Cannot allocate Pk_eq for matrix_vg.'
      error stop 1
    end if
    Hk_eq = C_0
    Pk_eq = C_0

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, ir, a, kdotR, ph0, &
    !$OMP                                  H_full, D_full, dH_full, P_full, tmp_proj)
    allocate(H_full(nwann, nwann), D_full(nwann, nwann, 3))
    allocate(dH_full(nwann, nwann, 3), P_full(nwann, nwann, 3))
    allocate(tmp_proj(n_trunc, n_trunc))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        H_full = C_0
        D_full = C_0
        dH_full = C_0

        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          ph0 = exp(C_I * kdotR) / real(ndegen(ir), dp)

          H_full = H_full + ph0 * Hmn_R(:,:,ir)
          do a = 1, 3
            D_full(:,:,a) = D_full(:,:,a) + ph0 * rmn_R(:,:,a,ir)
            dH_full(:,:,a) = dH_full(:,:,a) &
              + (C_I * Rvec_cart(a,ir) * ph0) * Hmn_R(:,:,ir)
          end do
        end do

        H_full = 0.5_dp * (H_full + conjg(transpose(H_full)))
        call project_to_trunc_withU(H_full, U_trunc(:,:,ikx,iky), nwann, n_trunc, tmp_proj)
        Hk_eq(:,:,ikx,iky) = tmp_proj

        do a = 1, 3
          D_full(:,:,a) = 0.5_dp * (D_full(:,:,a) + conjg(transpose(D_full(:,:,a))))
          dH_full(:,:,a) = 0.5_dp * (dH_full(:,:,a) + conjg(transpose(dH_full(:,:,a))))
          P_full(:,:,a) = dH_full(:,:,a) - C_I * &
            (matmul(D_full(:,:,a), H_full) - matmul(H_full, D_full(:,:,a)))
          P_full(:,:,a) = 0.5_dp * (P_full(:,:,a) + conjg(transpose(P_full(:,:,a))))
          call project_to_trunc_withU(P_full(:,:,a), U_trunc(:,:,ikx,iky), &
                                      nwann, n_trunc, tmp_proj)
          Pk_eq(:,:,a,ikx,iky) = tmp_proj
        end do
      end do
    end do
    !$OMP END DO

    deallocate(H_full, D_full, dH_full, P_full, tmp_proj)
    !$OMP END PARALLEL

    write(*,'(A)') '  Pre-computed matrix-VG matrices (H_eq, P_eq) using rmn_R.'
  end subroutine precompute_matrix_vg_matrices

  subroutine diagnose_pcenter_velocity(summary_file, kresolved_file)
    character(*), intent(in) :: summary_file, kresolved_file
    integer :: ikx, iky, ir, a, m, n, ir0, uk, us
    real(dp) :: kdotR
    complex(dp) :: ph0
    real(dp) :: centers(nwann, 3)
    real(dp), allocatable :: delta_center(:,:,:)
    complex(dp), allocatable :: H_full(:,:), D_full(:,:,:)
    complex(dp), allocatable :: dH_full(:,:,:), P_full(:,:,:), P_center(:,:,:)
    complex(dp), allocatable :: P_full_proj(:,:), P_center_proj(:,:), diff_proj(:,:)
    real(dp) :: eta_w, eta_t, eta_oo, eta_oe, eta_ee
    real(dp) :: nf_w, nd_w, nf_t, nd_t, nf_blk, nd_blk
    real(dp) :: sum_eta_w(3), sum_eta_t(3), sum_eta_oo(3), sum_eta_oe(3), sum_eta_ee(3)
    real(dp) :: max_eta_w(3), max_eta_t(3), max_eta_oo(3), max_eta_oe(3), max_eta_ee(3)
    integer :: nk_total

    if (.not. has_rmn) then
      write(*,*) 'ERROR: Pcenter diagnostic requires rmn_R from tb.dat or _r.dat.'
      error stop 1
    end if

    ir0 = find_R0()
    if (ir0 <= 0) then
      write(*,*) 'ERROR: cannot find R=0 for Wannier centers.'
      error stop 1
    end if

    do a = 1, 3
      do m = 1, nwann
        centers(m, a) = real(rmn_R(m, m, a, ir0), dp)
      end do
    end do

    allocate(delta_center(nwann, nwann, 3))
    do a = 1, 3
      do n = 1, nwann
        do m = 1, nwann
          delta_center(m, n, a) = centers(n, a) - centers(m, a)
        end do
      end do
    end do

    allocate(H_full(nwann, nwann), D_full(nwann, nwann, 3))
    allocate(dH_full(nwann, nwann, 3), P_full(nwann, nwann, 3), P_center(nwann, nwann, 3))
    allocate(P_full_proj(n_trunc, n_trunc), P_center_proj(n_trunc, n_trunc))
    allocate(diff_proj(n_trunc, n_trunc))

    sum_eta_w = 0.0_dp;  sum_eta_t = 0.0_dp
    sum_eta_oo = 0.0_dp; sum_eta_oe = 0.0_dp; sum_eta_ee = 0.0_dp
    max_eta_w = 0.0_dp;  max_eta_t = 0.0_dp
    max_eta_oo = 0.0_dp; max_eta_oe = 0.0_dp; max_eta_ee = 0.0_dp
    nk_total = nkx * nky

    open(newunit=uk, file=trim(kresolved_file), status='replace', action='write')
    write(uk,'(A)') '# ikx iky comp eta_wannier eta_trunc eta_oo eta_oe eta_ee norm_full_trunc norm_diff_trunc'

    do iky = 1, nky
      do ikx = 1, nkx
        H_full = C_0
        D_full = C_0
        dH_full = C_0
        P_center = C_0

        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          ph0 = exp(C_I * kdotR) / real(ndegen(ir), dp)

          H_full = H_full + ph0 * Hmn_R(:,:,ir)
          do a = 1, 3
            D_full(:,:,a) = D_full(:,:,a) + ph0 * rmn_R(:,:,a,ir)
            dH_full(:,:,a) = dH_full(:,:,a) &
              + (C_I * Rvec_cart(a,ir) * ph0) * Hmn_R(:,:,ir)
            P_center(:,:,a) = P_center(:,:,a) &
              + C_I * ph0 * (Rvec_cart(a,ir) + delta_center(:,:,a)) * Hmn_R(:,:,ir)
          end do
        end do

        H_full = 0.5_dp * (H_full + conjg(transpose(H_full)))

        do a = 1, 3
          D_full(:,:,a) = 0.5_dp * (D_full(:,:,a) + conjg(transpose(D_full(:,:,a))))
          dH_full(:,:,a) = 0.5_dp * (dH_full(:,:,a) + conjg(transpose(dH_full(:,:,a))))
          P_full(:,:,a) = dH_full(:,:,a) - C_I * &
            (matmul(D_full(:,:,a), H_full) - matmul(H_full, D_full(:,:,a)))
          P_full(:,:,a) = 0.5_dp * (P_full(:,:,a) + conjg(transpose(P_full(:,:,a))))
          P_center(:,:,a) = 0.5_dp * (P_center(:,:,a) + conjg(transpose(P_center(:,:,a))))

          nf_w = frob_norm(P_full(:,:,a))
          nd_w = frob_norm(P_full(:,:,a) - P_center(:,:,a))
          eta_w = safe_ratio(nd_w, nf_w)

          call project_to_trunc_withU(P_full(:,:,a), U_trunc(:,:,ikx,iky), &
                                      nwann, n_trunc, P_full_proj)
          call project_to_trunc_withU(P_center(:,:,a), U_trunc(:,:,ikx,iky), &
                                      nwann, n_trunc, P_center_proj)
          diff_proj = P_full_proj - P_center_proj

          nf_t = frob_norm(P_full_proj)
          nd_t = frob_norm(diff_proj)
          eta_t = safe_ratio(nd_t, nf_t)

          nf_blk = frob_norm(P_full_proj(1:nv, 1:nv))
          nd_blk = frob_norm(diff_proj(1:nv, 1:nv))
          eta_oo = safe_ratio(nd_blk, nf_blk)

          if (nv < n_trunc) then
            nf_blk = sqrt(frob_norm(P_full_proj(1:nv, nv+1:n_trunc))**2 + &
                          frob_norm(P_full_proj(nv+1:n_trunc, 1:nv))**2)
            nd_blk = sqrt(frob_norm(diff_proj(1:nv, nv+1:n_trunc))**2 + &
                          frob_norm(diff_proj(nv+1:n_trunc, 1:nv))**2)
            eta_oe = safe_ratio(nd_blk, nf_blk)

            nf_blk = frob_norm(P_full_proj(nv+1:n_trunc, nv+1:n_trunc))
            nd_blk = frob_norm(diff_proj(nv+1:n_trunc, nv+1:n_trunc))
            eta_ee = safe_ratio(nd_blk, nf_blk)
          else
            eta_oe = 0.0_dp
            eta_ee = 0.0_dp
          end if

          sum_eta_w(a) = sum_eta_w(a) + eta_w
          sum_eta_t(a) = sum_eta_t(a) + eta_t
          sum_eta_oo(a) = sum_eta_oo(a) + eta_oo
          sum_eta_oe(a) = sum_eta_oe(a) + eta_oe
          sum_eta_ee(a) = sum_eta_ee(a) + eta_ee
          max_eta_w(a) = max(max_eta_w(a), eta_w)
          max_eta_t(a) = max(max_eta_t(a), eta_t)
          max_eta_oo(a) = max(max_eta_oo(a), eta_oo)
          max_eta_oe(a) = max(max_eta_oe(a), eta_oe)
          max_eta_ee(a) = max(max_eta_ee(a), eta_ee)

          write(uk,'(3I6,7ES18.8)') ikx, iky, a, eta_w, eta_t, eta_oo, eta_oe, eta_ee, nf_t, nd_t
        end do
      end do
    end do

    close(uk)

    open(newunit=us, file=trim(summary_file), status='replace', action='write')
    write(us,'(A)') '# P_full vs P_center diagnostic'
    write(us,'(A)') '# P_full = dH/dk - i[D,H], using full rmn_R as in matrix_vg.'
    write(us,'(A)') '# P_center = i sum_R exp(ikR) (R + rbar_n - rbar_m) H(R).'
    write(us,'(A,I0)') '# nk_total = ', nk_total
    write(us,'(A,I0,A,I0)') '# band window = ', nb_start, ' to ', nb_end
    write(us,'(A,I0)') '# nv = ', nv
    write(us,'(A)') '# comp mean_eta_wannier max_eta_wannier mean_eta_trunc max_eta_trunc ' // &
                    'mean_eta_oo max_eta_oo mean_eta_oe max_eta_oe mean_eta_ee max_eta_ee'
    do a = 1, 3
      write(us,'(I6,10ES18.8)') a, &
        sum_eta_w(a)/real(nk_total,dp), max_eta_w(a), &
        sum_eta_t(a)/real(nk_total,dp), max_eta_t(a), &
        sum_eta_oo(a)/real(nk_total,dp), max_eta_oo(a), &
        sum_eta_oe(a)/real(nk_total,dp), max_eta_oe(a), &
        sum_eta_ee(a)/real(nk_total,dp), max_eta_ee(a)
    end do
    close(us)

    write(*,'(A)') '  Pcenter diagnostic complete.'
    write(*,'(A,A)') '  Summary:    ', trim(summary_file)
    write(*,'(A,A)') '  k-resolved: ', trim(kresolved_file)
    do a = 1, 2
      write(*,'(A,I0,A,ES12.4,A,ES12.4)') '  comp ', a, &
        ' mean eta_trunc=', sum_eta_t(a)/real(nk_total,dp), &
        ' max eta_trunc=', max_eta_t(a)
    end do

    deallocate(delta_center, H_full, D_full, dH_full, P_full, P_center)
    deallocate(P_full_proj, P_center_proj, diff_proj)
  end subroutine diagnose_pcenter_velocity

  real(dp) function frob_norm(M) result(val)
    complex(dp), intent(in) :: M(:,:)
    val = sqrt(sum(abs(M)**2))
  end function frob_norm

  real(dp) function safe_ratio(num, den) result(val)
    real(dp), intent(in) :: num, den
    if (den > 0.0_dp) then
      val = num / den
    else
      val = 0.0_dp
    end if
  end function safe_ratio

  subroutine propagate(do_decompose)
    logical, intent(in), optional :: do_decompose
    logical :: decompose, do_valley, gauge_peierls_vg, gauge_matrix_vg
    logical :: gauge_lg, gauge_lg_cov
    logical :: advance_step
    integer :: gauge_mode
    integer  :: it, ikx, iky, a, m, n, info_d
    real(dp) :: Jx_it, Jy_it, Nk_inv
    real(dp) :: Jx_intra_it, Jy_intra_it, Jx_inter_it, Jy_inter_it
    real(dp) :: Jx_K_it, Jy_K_it, Jx_Kp_it, Jy_Kp_it
    real(dp) :: E_now(3), E_mid(3), E_next(3)
    real(dp) :: A_now(3), A_mid(3), A_next(3)
    complex(dp) :: tr_val, tr_intra, tr_inter
    complex(dp), allocatable :: phase_A(:), phase_A_mid(:), phase_A_next(:)
    integer :: lwork_d, lrwork_d, liwork_d

    complex(dp), allocatable :: Ht(:,:), Ht_mid(:,:), Ht_next(:,:), vk_a(:,:)
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
    gauge_peierls_vg = (trim(gauge_method) == 'vg')
    gauge_matrix_vg = (trim(gauge_method) == 'matrix_vg')
    gauge_lg = (trim(gauge_method) == 'lg')
    gauge_lg_cov = (trim(gauge_method) == 'lg_cov' .or. &
                    trim(gauge_method) == 'houston_lg')

    if (gauge_peierls_vg) then
      gauge_mode = MODE_PEIERLS_VG
    else if (gauge_matrix_vg) then
      gauge_mode = MODE_MATRIX_VG
    else if (gauge_lg) then
      gauge_mode = MODE_LG
    else if (gauge_lg_cov) then
      gauge_mode = MODE_LG_COV
    else
      write(*,*) 'ERROR: gauge_method must be "vg", "matrix_vg", "lg", or "lg_cov", got: ', trim(gauge_method)
      error stop 1
    end if

    if (gauge_peierls_vg .and. .not. allocated(HR_proj)) then
      write(*,*) 'ERROR: call precompute_projected_matrices() before propagate()'
      error stop 1
    end if

    if (gauge_matrix_vg .and. (.not. allocated(Hk_eq) .or. .not. allocated(Pk_eq))) then
      write(*,*) 'ERROR: call precompute_matrix_vg_matrices() before propagate()'
      error stop 1
    end if

    if ((gauge_lg .or. gauge_lg_cov) .and. .not. allocated(Hk_eq)) then
      write(*,*) 'ERROR: call precompute_lg_matrices() before propagate() in LG mode'
      error stop 1
    end if
    if ((gauge_lg .or. gauge_lg_cov) .and. (.not. allocated(Dk_eq) .or. .not. allocated(Pk_eq))) then
      write(*,*) 'ERROR: LG mode requires Dk_eq and Pk_eq current matrices.'
      error stop 1
    end if

    if (gauge_lg_cov) then
      call propagate_lg_covariant(decompose)
      return
    end if

    Nk_inv = 1.0_dp / (real(nkx * nky, dp) * A_cell)
    allocate(phase_A(nrpts), phase_A_mid(nrpts), phase_A_next(nrpts))
    phase_A = C_1
    phase_A_mid = C_1
    phase_A_next = C_1

    lwork_d  = 2 * n_trunc + n_trunc * n_trunc
    lrwork_d = 1 + 5 * n_trunc + 2 * n_trunc * n_trunc
    liwork_d = 3 + 5 * n_trunc

    do it = 1, nt
      Jx_it = 0.0_dp;       Jy_it = 0.0_dp
      Jx_intra_it = 0.0_dp; Jy_intra_it = 0.0_dp
      Jx_inter_it = 0.0_dp; Jy_inter_it = 0.0_dp
      Jx_K_it = 0.0_dp;     Jy_K_it = 0.0_dp
      Jx_Kp_it = 0.0_dp;    Jy_Kp_it = 0.0_dp
      advance_step = (it < nt)

      E_now = Et_vec(it, :)
      A_now = At_vec(it, :)
      if (advance_step) then
        E_mid = 0.5_dp * (Et_vec(it, :) + Et_vec(it+1, :))
        E_next = Et_vec(it+1, :)
        A_mid = 0.5_dp * (At_vec(it, :) + At_vec(it+1, :))
        A_next = At_vec(it+1, :)
      else
        E_mid = E_now
        E_next = E_now
        A_mid = A_now
        A_next = A_now
      end if

      if (gauge_peierls_vg) then
        call compute_phase_A(A_now, phase_A)
        if (advance_step) then
          call compute_phase_A(A_mid, phase_A_mid)
          call compute_phase_A(A_next, phase_A_next)
        else
          phase_A_mid = phase_A
          phase_A_next = phase_A
        end if
      end if

      !$OMP PARALLEL DEFAULT(shared) &
      !$OMP PRIVATE(ikx, iky, a, m, n, tr_val, tr_intra, tr_inter, info_d, &
      !$OMP         Ht, Ht_mid, Ht_next, vk_a, rho_k, rho_new, &
      !$OMP         k1, k2, k3, k4, rho_tmp, AB, &
      !$OMP         Wmat, eig_tmp, rho_band, v_band, tmp_mat, &
      !$OMP         work_d, rwork_d, iwork_d) &
      !$OMP REDUCTION(+:Jx_it, Jy_it, &
      !$OMP           Jx_intra_it, Jy_intra_it, Jx_inter_it, Jy_inter_it, &
      !$OMP           Jx_K_it, Jy_K_it, Jx_Kp_it, Jy_Kp_it)

      allocate(Ht(n_trunc,n_trunc), Ht_mid(n_trunc,n_trunc), Ht_next(n_trunc,n_trunc))
      allocate(vk_a(n_trunc,n_trunc))
      allocate(rho_k(n_trunc,n_trunc), rho_new(n_trunc,n_trunc))
      allocate(k1(n_trunc,n_trunc), k2(n_trunc,n_trunc))
      allocate(k3(n_trunc,n_trunc), k4(n_trunc,n_trunc))
      allocate(rho_tmp(n_trunc,n_trunc), AB(n_trunc,n_trunc))
      allocate(Wmat(n_trunc,n_trunc), eig_tmp(n_trunc))
      allocate(rho_band(n_trunc,n_trunc), tmp_mat(n_trunc,n_trunc))
      allocate(work_d(lwork_d), rwork_d(lrwork_d), iwork_d(liwork_d))
      if (decompose) then
        allocate(v_band(n_trunc,n_trunc))
      end if

      !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
      do iky = 1, nky
        do ikx = 1, nkx

          ! Current is evaluated at the labelled time t=(it-1)*dt,
          ! before advancing rho to the next time node.
          rho_k = rho(:,:,ikx,iky)
          call build_hamiltonian(ikx, iky, E_now, A_now, phase_A, gauge_mode, Ht)

          ! --- Diagonalize for intra/inter decomposition of J(t) ---
          if (decompose) then
            Wmat = Ht
            call zheevd('V', 'U', n_trunc, Wmat, n_trunc, eig_tmp, &
                        work_d, lwork_d, rwork_d, lrwork_d, iwork_d, liwork_d, info_d)
            if (info_d /= 0) then
              write(*,*) 'ERROR: zheevd failed during decomposition at k-point', ikx, iky, ' info=', info_d
              error stop 1
            end if
            call zgemm('C','N',n_trunc,n_trunc,n_trunc,C_1, &
                       Wmat,n_trunc,rho_k,n_trunc,C_0,tmp_mat,n_trunc)
            call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1, &
                       tmp_mat,n_trunc,Wmat,n_trunc,C_0,rho_band,n_trunc)
          end if

          ! --- Current (gauge-dependent velocity) ---
          do a = 1, 2
            call build_velocity_component(ikx, iky, a, A_now, phase_A, gauge_mode, vk_a)

            tr_val = C_0
            do n = 1, n_trunc
              do m = 1, n_trunc
                tr_val = tr_val + vk_a(m,n) * rho_k(n,m)
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

          if (advance_step) then
            ! Time-dependent RK4: H is evaluated at t, t+dt/2, and t+dt.
            call commutator_rhs(Ht, rho_k, k1, AB)

            call build_hamiltonian(ikx, iky, E_mid, A_mid, phase_A_mid, gauge_mode, Ht_mid)
            rho_tmp = rho_k + 0.5_dp * dt * k1
            call commutator_rhs(Ht_mid, rho_tmp, k2, AB)

            rho_tmp = rho_k + 0.5_dp * dt * k2
            call commutator_rhs(Ht_mid, rho_tmp, k3, AB)

            call build_hamiltonian(ikx, iky, E_next, A_next, phase_A_next, gauge_mode, Ht_next)
            rho_tmp = rho_k + dt * k3
            call commutator_rhs(Ht_next, rho_tmp, k4, AB)

            rho_new = rho_k + (dt / 6.0_dp) * (k1 + 2.0_dp*k2 + 2.0_dp*k3 + k4)

            ! --- Dephasing ---
            if (mod(it, n_dt_deph) == 0) then
              call apply_instantaneous_dephasing(Ht_next, rho_new, Wmat, eig_tmp, &
                                                 tmp_mat, rho_band, work_d, rwork_d, &
                                                 iwork_d, info_d)
            end if

            rho(:,:,ikx,iky) = rho_new
          end if

        end do
      end do
      !$OMP END DO

      deallocate(Ht, Ht_mid, Ht_next, vk_a, rho_k, rho_new)
      deallocate(k1, k2, k3, k4, rho_tmp, AB)
      if (decompose) deallocate(v_band)
      deallocate(Wmat, eig_tmp, rho_band, tmp_mat)
      deallocate(work_d, rwork_d, iwork_d)
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

    deallocate(phase_A, phase_A_mid, phase_A_next)
  end subroutine propagate

  subroutine propagate_lg_covariant(do_decompose)
    logical, intent(in) :: do_decompose
    logical :: decompose, do_valley, advance_step
    integer :: it, ikx, iky, a, m, n, info_d
    real(dp) :: Nk_inv
    real(dp) :: Jx_it, Jy_it
    real(dp) :: Jx_intra_it, Jy_intra_it, Jx_inter_it, Jy_inter_it
    real(dp) :: Jx_K_it, Jy_K_it, Jx_Kp_it, Jy_Kp_it
    real(dp) :: E_now(3), E_mid(3), E_next(3)
    complex(dp) :: tr_val, tr_intra, tr_inter
    integer :: lwork_d, lrwork_d, liwork_d
    complex(dp), allocatable :: k1(:,:,:,:), k2(:,:,:,:)
    complex(dp), allocatable :: k3(:,:,:,:), k4(:,:,:,:)
    complex(dp), allocatable :: rho_stage(:,:,:,:), rho_new(:,:,:,:)
    complex(dp), allocatable :: Ht(:,:), Ht_next(:,:), vk_a(:,:)
    complex(dp), allocatable :: Wmat(:,:), rho_band(:,:), v_band(:,:), tmp_mat(:,:)
    real(dp),    allocatable :: eig_tmp(:)
    complex(dp), allocatable :: work_d(:)
    real(dp),    allocatable :: rwork_d(:)
    integer,     allocatable :: iwork_d(:)

    write(*,'(A)') '  Using gauge-covariant LG propagation:'
    write(*,'(A)') '    d rho/dt = -i[H0 + E.D, rho] + E.D_k rho'
    write(*,'(A)') '    D_k rho uses parallel-transported finite differences.'

    decompose = do_decompose
    do_valley = allocated(valley_id) .and. decompose
    Nk_inv = 1.0_dp / (real(nkx * nky, dp) * A_cell)

    lwork_d  = 2 * n_trunc + n_trunc * n_trunc
    lrwork_d = 1 + 5 * n_trunc + 2 * n_trunc * n_trunc
    liwork_d = 3 + 5 * n_trunc

    allocate(k1(n_trunc,n_trunc,nkx,nky), k2(n_trunc,n_trunc,nkx,nky))
    allocate(k3(n_trunc,n_trunc,nkx,nky), k4(n_trunc,n_trunc,nkx,nky))
    allocate(rho_stage(n_trunc,n_trunc,nkx,nky))
    allocate(rho_new(n_trunc,n_trunc,nkx,nky))

    do it = 1, nt
      Jx_it = 0.0_dp;       Jy_it = 0.0_dp
      Jx_intra_it = 0.0_dp; Jy_intra_it = 0.0_dp
      Jx_inter_it = 0.0_dp; Jy_inter_it = 0.0_dp
      Jx_K_it = 0.0_dp;     Jy_K_it = 0.0_dp
      Jx_Kp_it = 0.0_dp;    Jy_Kp_it = 0.0_dp
      advance_step = (it < nt)

      E_now = Et_vec(it, :)
      if (advance_step) then
        E_mid = 0.5_dp * (Et_vec(it, :) + Et_vec(it+1, :))
        E_next = Et_vec(it+1, :)
      else
        E_mid = E_now
        E_next = E_now
      end if

      !$OMP PARALLEL DEFAULT(shared) &
      !$OMP PRIVATE(ikx, iky, a, m, n, tr_val, tr_intra, tr_inter, info_d, &
      !$OMP         Ht, vk_a, Wmat, eig_tmp, rho_band, v_band, tmp_mat, &
      !$OMP         work_d, rwork_d, iwork_d) &
      !$OMP REDUCTION(+:Jx_it, Jy_it, &
      !$OMP           Jx_intra_it, Jy_intra_it, Jx_inter_it, Jy_inter_it, &
      !$OMP           Jx_K_it, Jy_K_it, Jx_Kp_it, Jy_Kp_it)

      allocate(Ht(n_trunc,n_trunc), vk_a(n_trunc,n_trunc))
      allocate(Wmat(n_trunc,n_trunc), eig_tmp(n_trunc))
      allocate(rho_band(n_trunc,n_trunc), tmp_mat(n_trunc,n_trunc))
      allocate(work_d(lwork_d), rwork_d(lrwork_d), iwork_d(liwork_d))
      if (decompose) allocate(v_band(n_trunc,n_trunc))

      !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
      do iky = 1, nky
        do ikx = 1, nkx
          call build_lg_hamiltonian(ikx, iky, E_now, Ht)

          if (decompose) then
            Wmat = Ht
            call zheevd('V', 'U', n_trunc, Wmat, n_trunc, eig_tmp, &
                        work_d, lwork_d, rwork_d, lrwork_d, iwork_d, liwork_d, info_d)
            if (info_d /= 0) then
              write(*,*) 'ERROR: zheevd failed during LG-cov decomposition at k-point', &
                         ikx, iky, ' info=', info_d
              error stop 1
            end if
            call zgemm('C','N',n_trunc,n_trunc,n_trunc,C_1, &
                       Wmat,n_trunc,rho(:,:,ikx,iky),n_trunc,C_0,tmp_mat,n_trunc)
            call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1, &
                       tmp_mat,n_trunc,Wmat,n_trunc,C_0,rho_band,n_trunc)
          end if

          do a = 1, 2
            vk_a = Pk_eq(:,:,a,ikx,iky)

            tr_val = C_0
            do n = 1, n_trunc
              do m = 1, n_trunc
                tr_val = tr_val + vk_a(m,n) * rho(n,m,ikx,iky)
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

      deallocate(Ht, vk_a, Wmat, eig_tmp, rho_band, tmp_mat)
      deallocate(work_d, rwork_d, iwork_d)
      if (decompose) deallocate(v_band)
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

      if (advance_step) then
        call lg_covariant_rhs_all(E_now, rho, k1)
        rho_stage = rho + 0.5_dp * dt * k1

        call lg_covariant_rhs_all(E_mid, rho_stage, k2)
        rho_stage = rho + 0.5_dp * dt * k2

        call lg_covariant_rhs_all(E_mid, rho_stage, k3)
        rho_stage = rho + dt * k3

        call lg_covariant_rhs_all(E_next, rho_stage, k4)
        rho_new = rho + (dt / 6.0_dp) * (k1 + 2.0_dp*k2 + 2.0_dp*k3 + k4)

        !$OMP PARALLEL DEFAULT(shared) &
        !$OMP PRIVATE(ikx, iky, info_d, Ht_next, Wmat, eig_tmp, tmp_mat, rho_band, &
        !$OMP         work_d, rwork_d, iwork_d)
        allocate(Ht_next(n_trunc,n_trunc), Wmat(n_trunc,n_trunc), eig_tmp(n_trunc))
        allocate(tmp_mat(n_trunc,n_trunc), rho_band(n_trunc,n_trunc))
        allocate(work_d(lwork_d), rwork_d(lrwork_d), iwork_d(liwork_d))

        !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
        do iky = 1, nky
          do ikx = 1, nkx
            if (mod(it, n_dt_deph) == 0) then
              call build_lg_hamiltonian(ikx, iky, E_next, Ht_next)
              call apply_instantaneous_dephasing(Ht_next, rho_new(:,:,ikx,iky), &
                                                 Wmat, eig_tmp, tmp_mat, rho_band, &
                                                 work_d, rwork_d, iwork_d, info_d)
            else
              rho_new(:,:,ikx,iky) = 0.5_dp * &
                (rho_new(:,:,ikx,iky) + conjg(transpose(rho_new(:,:,ikx,iky))))
            end if
          end do
        end do
        !$OMP END DO

        deallocate(Ht_next, Wmat, eig_tmp, tmp_mat, rho_band)
        deallocate(work_d, rwork_d, iwork_d)
        !$OMP END PARALLEL

        rho = rho_new
      end if

      if (mod(it, max(nt/20, 1)) == 0) then
        write(*,'(A,I0,A,I0,A,F6.1,A)') '  Step ', it, ' / ', nt, &
          ' (', 100.0_dp*real(it,dp)/real(nt,dp), '%)'
      end if
    end do

    deallocate(k1, k2, k3, k4, rho_stage, rho_new)
  end subroutine propagate_lg_covariant

  subroutine lg_covariant_rhs_all(E_now, rho_in, rhs)
    real(dp),    intent(in)  :: E_now(3)
    complex(dp), intent(in)  :: rho_in(n_trunc, n_trunc, nkx, nky)
    complex(dp), intent(out) :: rhs(n_trunc, n_trunc, nkx, nky)
    integer :: ikx, iky
    complex(dp), allocatable :: Ht(:,:), AB(:,:)

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, Ht, AB)
    allocate(Ht(n_trunc,n_trunc), AB(n_trunc,n_trunc))

    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        call build_lg_hamiltonian(ikx, iky, E_now, Ht)
        call commutator_rhs(Ht, rho_in(:,:,ikx,iky), rhs(:,:,ikx,iky), AB)
        call add_lg_covariant_gradient(ikx, iky, E_now, rho_in, rhs(:,:,ikx,iky))
      end do
    end do
    !$OMP END DO

    deallocate(Ht, AB)
    !$OMP END PARALLEL
  end subroutine lg_covariant_rhs_all

  subroutine add_lg_covariant_gradient(ikx, iky, E_now, rho_in, rhs_k)
    integer,     intent(in)    :: ikx, iky
    real(dp),    intent(in)    :: E_now(3)
    complex(dp), intent(in)    :: rho_in(n_trunc, n_trunc, nkx, nky)
    complex(dp), intent(inout) :: rhs_k(n_trunc, n_trunc)
    real(dp) :: det_b, coef_f1, coef_f2
    complex(dp), allocatable :: drho_df1(:,:), drho_df2(:,:)

    if (abs(E_now(1)) < 1.0e-30_dp .and. abs(E_now(2)) < 1.0e-30_dp) return

    det_b = b1(1) * b2(2) - b1(2) * b2(1)
    if (abs(det_b) < 1.0e-18_dp) then
      write(*,*) 'ERROR: singular in-plane reciprocal basis for LG covariant derivative.'
      error stop 1
    end if

    coef_f1 = ( E_now(1) * b2(2) - E_now(2) * b2(1)) / det_b
    coef_f2 = (-E_now(1) * b1(2) + E_now(2) * b1(1)) / det_b

    allocate(drho_df1(n_trunc,n_trunc), drho_df2(n_trunc,n_trunc))
    call finite_diff_rho_fractional(1, ikx, iky, rho_in, drho_df1)
    call finite_diff_rho_fractional(2, ikx, iky, rho_in, drho_df2)
    rhs_k = rhs_k + coef_f1 * drho_df1 + coef_f2 * drho_df2
    deallocate(drho_df1, drho_df2)
  end subroutine add_lg_covariant_gradient

  subroutine finite_diff_rho_fractional(dir, ikx, iky, rho_in, deriv)
    integer,     intent(in)  :: dir, ikx, iky
    complex(dp), intent(in)  :: rho_in(n_trunc, n_trunc, nkx, nky)
    complex(dp), intent(out) :: deriv(n_trunc, n_trunc)
    integer :: im2, im1, ip1, ip2
    real(dp) :: df

    if (dir == 1) then
      df = 1.0_dp / real(nkx, dp)
      im2 = wrap_index(ikx - 2, nkx)
      im1 = wrap_index(ikx - 1, nkx)
      ip1 = wrap_index(ikx + 1, nkx)
      ip2 = wrap_index(ikx + 2, nkx)
      if (nkx >= 5) then
        call finite_diff_rho_parallel(ikx, iky, im2, iky, im1, iky, &
                                      ip1, iky, ip2, iky, rho_in, deriv, df)
      else if (nkx >= 3) then
        call finite_diff_rho_parallel2(ikx, iky, im1, iky, ip1, iky, &
                                       rho_in, deriv, df)
      else
        deriv = C_0
      end if
    else
      df = 1.0_dp / real(nky, dp)
      im2 = wrap_index(iky - 2, nky)
      im1 = wrap_index(iky - 1, nky)
      ip1 = wrap_index(iky + 1, nky)
      ip2 = wrap_index(iky + 2, nky)
      if (nky >= 5) then
        call finite_diff_rho_parallel(ikx, iky, ikx, im2, ikx, im1, &
                                      ikx, ip1, ikx, ip2, rho_in, deriv, df)
      else if (nky >= 3) then
        call finite_diff_rho_parallel2(ikx, iky, ikx, im1, ikx, ip1, &
                                       rho_in, deriv, df)
      else
        deriv = C_0
      end if
    end if
  end subroutine finite_diff_rho_fractional

  subroutine finite_diff_rho_parallel(ikx0, iky0, im2x, im2y, im1x, im1y, &
                                      ip1x, ip1y, ip2x, ip2y, rho_in, deriv, df)
    integer,     intent(in)  :: ikx0, iky0
    integer,     intent(in)  :: im2x, im2y, im1x, im1y, ip1x, ip1y, ip2x, ip2y
    complex(dp), intent(in)  :: rho_in(n_trunc, n_trunc, nkx, nky)
    complex(dp), intent(out) :: deriv(n_trunc, n_trunc)
    real(dp),    intent(in)  :: df
    complex(dp), allocatable :: r_im2(:,:), r_im1(:,:), r_ip1(:,:), r_ip2(:,:)

    allocate(r_im2(n_trunc,n_trunc), r_im1(n_trunc,n_trunc))
    allocate(r_ip1(n_trunc,n_trunc), r_ip2(n_trunc,n_trunc))
    call transport_density_to_k(ikx0, iky0, im2x, im2y, rho_in(:,:,im2x,im2y), r_im2)
    call transport_density_to_k(ikx0, iky0, im1x, im1y, rho_in(:,:,im1x,im1y), r_im1)
    call transport_density_to_k(ikx0, iky0, ip1x, ip1y, rho_in(:,:,ip1x,ip1y), r_ip1)
    call transport_density_to_k(ikx0, iky0, ip2x, ip2y, rho_in(:,:,ip2x,ip2y), r_ip2)
    deriv = (r_im2 - 8.0_dp*r_im1 + 8.0_dp*r_ip1 - r_ip2) / (12.0_dp * df)
    deallocate(r_im2, r_im1, r_ip1, r_ip2)
  end subroutine finite_diff_rho_parallel

  subroutine finite_diff_rho_parallel2(ikx0, iky0, im1x, im1y, ip1x, ip1y, &
                                       rho_in, deriv, df)
    integer,     intent(in)  :: ikx0, iky0, im1x, im1y, ip1x, ip1y
    complex(dp), intent(in)  :: rho_in(n_trunc, n_trunc, nkx, nky)
    complex(dp), intent(out) :: deriv(n_trunc, n_trunc)
    real(dp),    intent(in)  :: df
    complex(dp), allocatable :: r_im1(:,:), r_ip1(:,:)

    allocate(r_im1(n_trunc,n_trunc), r_ip1(n_trunc,n_trunc))
    call transport_density_to_k(ikx0, iky0, im1x, im1y, rho_in(:,:,im1x,im1y), r_im1)
    call transport_density_to_k(ikx0, iky0, ip1x, ip1y, rho_in(:,:,ip1x,ip1y), r_ip1)
    deriv = (r_ip1 - r_im1) / (2.0_dp * df)
    deallocate(r_im1, r_ip1)
  end subroutine finite_diff_rho_parallel2

  subroutine transport_density_to_k(ikx0, iky0, ikx1, iky1, rho_src, rho_dst)
    integer,     intent(in)  :: ikx0, iky0, ikx1, iky1
    complex(dp), intent(in)  :: rho_src(n_trunc, n_trunc)
    complex(dp), intent(out) :: rho_dst(n_trunc, n_trunc)
    complex(dp), allocatable :: S(:,:), tmp(:,:)

    if (ikx0 == ikx1 .and. iky0 == iky1) then
      rho_dst = rho_src
      return
    end if

    allocate(S(n_trunc,n_trunc), tmp(n_trunc,n_trunc))
    call zgemm('C','N', n_trunc, n_trunc, nwann, C_1, &
               U_trunc(:,:,ikx0,iky0), nwann, U_trunc(:,:,ikx1,iky1), &
               nwann, C_0, S, n_trunc)
    call zgemm('N','N', n_trunc, n_trunc, n_trunc, C_1, &
               S, n_trunc, rho_src, n_trunc, C_0, tmp, n_trunc)
    call zgemm('N','C', n_trunc, n_trunc, n_trunc, C_1, &
               tmp, n_trunc, S, n_trunc, C_0, rho_dst, n_trunc)
    rho_dst = 0.5_dp * (rho_dst + conjg(transpose(rho_dst)))
    deallocate(S, tmp)
  end subroutine transport_density_to_k

  integer function wrap_index(i, n) result(idx)
    integer, intent(in) :: i, n
    idx = modulo(i - 1, n) + 1
  end function wrap_index

  subroutine apply_instantaneous_dephasing(Ht_deph, rho_inout, Wmat, eig_tmp, &
                                           tmp_mat, rho_band, work_d, rwork_d, &
                                           iwork_d, info_d)
    complex(dp), intent(in)    :: Ht_deph(n_trunc, n_trunc)
    complex(dp), intent(inout) :: rho_inout(n_trunc, n_trunc)
    complex(dp), intent(inout) :: Wmat(n_trunc, n_trunc)
    complex(dp), intent(inout) :: tmp_mat(n_trunc, n_trunc)
    complex(dp), intent(inout) :: rho_band(n_trunc, n_trunc)
    complex(dp), intent(inout) :: work_d(:)
    real(dp),    intent(inout) :: eig_tmp(n_trunc)
    real(dp),    intent(inout) :: rwork_d(:)
    integer,     intent(inout) :: iwork_d(:)
    integer,     intent(out)   :: info_d
    integer :: m, n

    Wmat = Ht_deph
    call zheevd('V', 'U', n_trunc, Wmat, n_trunc, eig_tmp, &
                work_d, size(work_d), rwork_d, size(rwork_d), &
                iwork_d, size(iwork_d), info_d)
    if (info_d /= 0) then
      write(*,*) 'ERROR: zheevd failed during instantaneous dephasing. info=', info_d
      error stop 1
    end if

    call zgemm('C','N', n_trunc, n_trunc, n_trunc, C_1, &
               Wmat, n_trunc, rho_inout, n_trunc, C_0, tmp_mat, n_trunc)
    call zgemm('N','N', n_trunc, n_trunc, n_trunc, C_1, &
               tmp_mat, n_trunc, Wmat, n_trunc, C_0, rho_band, n_trunc)

    do n = 1, n_trunc
      do m = 1, n_trunc
        if (m /= n) rho_band(m,n) = rho_band(m,n) * deph_factor
      end do
    end do

    call zgemm('N','N', n_trunc, n_trunc, n_trunc, C_1, &
               Wmat, n_trunc, rho_band, n_trunc, C_0, tmp_mat, n_trunc)
    call zgemm('N','C', n_trunc, n_trunc, n_trunc, C_1, &
               tmp_mat, n_trunc, Wmat, n_trunc, C_0, rho_inout, n_trunc)

    rho_inout = 0.5_dp * (rho_inout + conjg(transpose(rho_inout)))
  end subroutine apply_instantaneous_dephasing

  subroutine build_hamiltonian(ikx, iky, E_now, A_now, phase_A, gauge_mode, Ht)
    integer,     intent(in)  :: ikx, iky
    real(dp),    intent(in)  :: E_now(3)
    real(dp),    intent(in)  :: A_now(3)
    complex(dp), intent(in)  :: phase_A(nrpts)
    integer,     intent(in)  :: gauge_mode
    complex(dp), intent(out) :: Ht(n_trunc, n_trunc)
    integer :: ir, a

    if (gauge_mode == MODE_PEIERLS_VG) then
      Ht = C_0
      do ir = 1, nrpts
        Ht = Ht + phase_A(ir) * HR_proj(:,:,ir,ikx,iky)
      end do
    else if (gauge_mode == MODE_MATRIX_VG) then
      Ht = Hk_eq(:,:,ikx,iky)
      do a = 1, 3
        if (abs(A_now(a)) > 1.0e-30_dp) then
          Ht = Ht + A_now(a) * Pk_eq(:,:,a,ikx,iky)
        end if
      end do
    else
      call build_lg_hamiltonian(ikx, iky, E_now, Ht)
    end if
  end subroutine build_hamiltonian

  subroutine build_lg_hamiltonian(ikx, iky, E_now, Ht)
    integer,     intent(in)  :: ikx, iky
    real(dp),    intent(in)  :: E_now(3)
    complex(dp), intent(out) :: Ht(n_trunc, n_trunc)
    integer :: a

    Ht = Hk_eq(:,:,ikx,iky)
    do a = 1, 3
      if (abs(E_now(a)) > 1.0e-30_dp) then
        Ht = Ht + E_now(a) * Dk_eq(:,:,a,ikx,iky)
      end if
    end do
  end subroutine build_lg_hamiltonian

  subroutine build_velocity_component(ikx, iky, a, A_now, phase_A, gauge_mode, vk_a)
    integer,     intent(in)  :: ikx, iky, a
    real(dp),    intent(in)  :: A_now(3)
    complex(dp), intent(in)  :: phase_A(nrpts)
    integer,     intent(in)  :: gauge_mode
    complex(dp), intent(out) :: vk_a(n_trunc, n_trunc)
    integer :: ir, n

    if (gauge_mode == MODE_PEIERLS_VG) then
      vk_a = C_0
      do ir = 1, nrpts
        vk_a = vk_a + (C_I * Rvec_cart(a,ir) * phase_A(ir)) * HR_proj(:,:,ir,ikx,iky)
      end do
    else if (gauge_mode == MODE_MATRIX_VG) then
      vk_a = Pk_eq(:,:,a,ikx,iky)
      do n = 1, n_trunc
        vk_a(n,n) = vk_a(n,n) + A_now(a)
      end do
    else
      vk_a = Pk_eq(:,:,a,ikx,iky)
    end if
  end subroutine build_velocity_component

  subroutine commutator_rhs(Ht, rho_in, rhs, AB)
    complex(dp), intent(in)  :: Ht(n_trunc, n_trunc), rho_in(n_trunc, n_trunc)
    complex(dp), intent(out) :: rhs(n_trunc, n_trunc)
    complex(dp), intent(out) :: AB(n_trunc, n_trunc)

    call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Ht,n_trunc,rho_in,n_trunc,C_0,AB,n_trunc)
    call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,rho_in,n_trunc,Ht,n_trunc,C_0,rhs,n_trunc)
    rhs = -C_I * (AB - rhs)
  end subroutine commutator_rhs

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

  subroutine diagnose_tb_quality()
    integer :: ikx, iky, ir, a
    real(dp) :: kdotR
    real(dp) :: eps_H_max, eps_r_max(3), eps_v_max(3)
    real(dp) :: frob_H_sum, frob_r_sum(3), frob_v_sum(3)
    real(dp) :: antih_r_sum(3), antih_v_sum(3)
    real(dp) :: err_val, frob_val, antih_val
    integer  :: cnt
    complex(dp) :: ph0
    complex(dp), allocatable :: H_full(:,:), D_full(:,:,:)
    complex(dp), allocatable :: dH_full(:,:,:), P_full(:,:,:)
    complex(dp), allocatable :: diff(:,:)
    character(1), parameter :: axis_label(3) = ['x', 'y', 'z']

    if (.not. has_rmn) then
      write(*,'(A)') '  TB quality diagnostics skipped: no dipole matrix.'
      return
    end if

    eps_H_max = 0.0_dp; eps_r_max = 0.0_dp; eps_v_max = 0.0_dp
    frob_H_sum = 0.0_dp; frob_r_sum = 0.0_dp; frob_v_sum = 0.0_dp
    antih_r_sum = 0.0_dp; antih_v_sum = 0.0_dp
    cnt = 0

    allocate(H_full(nwann,nwann), D_full(nwann,nwann,3))
    allocate(dH_full(nwann,nwann,3), P_full(nwann,nwann,3))
    allocate(diff(nwann,nwann))

    do iky = 1, nky
      do ikx = 1, nkx
        H_full = C_0; D_full = C_0; dH_full = C_0

        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          ph0 = exp(C_I * kdotR) / real(ndegen(ir), dp)
          H_full = H_full + ph0 * Hmn_R(:,:,ir)
          do a = 1, 3
            D_full(:,:,a) = D_full(:,:,a) + ph0 * rmn_R(:,:,a,ir)
            dH_full(:,:,a) = dH_full(:,:,a) + (C_I * Rvec_cart(a,ir) * ph0) * Hmn_R(:,:,ir)
          end do
        end do

        diff = H_full - conjg(transpose(H_full))
        err_val = maxval(abs(diff))
        eps_H_max = max(eps_H_max, err_val)
        frob_val = sqrt(real(sum(abs(H_full)**2), dp))
        if (frob_val > 0.0_dp) frob_H_sum = frob_H_sum + err_val / frob_val

        do a = 1, 3
          diff = D_full(:,:,a) - conjg(transpose(D_full(:,:,a)))
          err_val = maxval(abs(diff))
          eps_r_max(a) = max(eps_r_max(a), err_val)
          antih_val = sqrt(real(sum(abs(diff)**2), dp))
          frob_val  = sqrt(real(sum(abs(D_full(:,:,a))**2), dp))
          if (frob_val > 0.0_dp) then
            frob_r_sum(a) = frob_r_sum(a) + antih_val / frob_val
          end if
          antih_r_sum(a) = antih_r_sum(a) + antih_val

          P_full(:,:,a) = dH_full(:,:,a) - C_I * &
            (matmul(D_full(:,:,a), H_full) - matmul(H_full, D_full(:,:,a)))
          diff = P_full(:,:,a) - conjg(transpose(P_full(:,:,a)))
          err_val = maxval(abs(diff))
          eps_v_max(a) = max(eps_v_max(a), err_val)
          antih_val = sqrt(real(sum(abs(diff)**2), dp))
          frob_val  = sqrt(real(sum(abs(P_full(:,:,a))**2), dp))
          if (frob_val > 0.0_dp) then
            frob_v_sum(a) = frob_v_sum(a) + antih_val / frob_val
          end if
          antih_v_sum(a) = antih_v_sum(a) + antih_val
        end do
        cnt = cnt + 1
      end do
    end do

    deallocate(H_full, D_full, dH_full, P_full, diff)

    write(*,'(A)') '==========================================='
    write(*,'(A)') '  TB Matrix Quality Diagnostics'
    write(*,'(A)') '==========================================='
    write(*,'(A,ES10.2,A,ES10.2)') '  H(k)   max|H-H+|=', eps_H_max, &
      '  avg rel_frob=', frob_H_sum / real(cnt, dp)
    do a = 1, 3
      write(*,'(A,A1,A,ES10.2,A,ES10.2,A,ES10.2)') &
        '  r_', axis_label(a), '(k) max|r-r+|=', eps_r_max(a), &
        '  avg |antiherm|_F=', antih_r_sum(a) / real(cnt, dp), &
        '  avg rel_frob=', frob_r_sum(a) / real(cnt, dp)
    end do
    do a = 1, 3
      write(*,'(A,A1,A,ES10.2,A,ES10.2,A,ES10.2)') &
        '  v_', axis_label(a), '(k) max|v-v+|=', eps_v_max(a), &
        '  avg |antiherm|_F=', antih_v_sum(a) / real(cnt, dp), &
        '  avg rel_frob=', frob_v_sum(a) / real(cnt, dp)
    end do
    write(*,'(A)') '==========================================='
  end subroutine diagnose_tb_quality

end module mod_sbe
