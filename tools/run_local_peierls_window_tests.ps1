$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo "hhg_sbe.exe"
if (-not (Test-Path $exe)) {
  $exe = Join-Path $repo "hhg_sbe"
}
if (-not (Test-Path $exe)) {
  throw "Cannot find hhg_sbe executable under $repo"
}

$tbFile = (Join-Path $repo "local_runs/CrI3_tb.dat").Replace("\", "/")
if (-not (Test-Path $tbFile)) {
  throw "Cannot find ASCII-path TB hardlink: $tbFile"
}

$root = Join-Path $repo "local_runs/peierls_window_20260517_nodeph"
New-Item -ItemType Directory -Force -Path $root | Out-Null

$env:PATH = "C:\msys64\mingw64\bin;C:\msys64\usr\bin;$env:PATH"

$cases = @(
  @{Name="k10_b20"; Nk=10; NbStart=75; NbEnd=94},
  @{Name="k10_b30"; Nk=10; NbStart=70; NbEnd=99},
  @{Name="k10_b40"; Nk=10; NbStart=65; NbEnd=104},
  @{Name="k10_b50"; Nk=10; NbStart=60; NbEnd=109},
  @{Name="k10_b60"; Nk=10; NbStart=53; NbEnd=112},
  @{Name="k20_b30"; Nk=20; NbStart=70; NbEnd=99},
  @{Name="k20_b40"; Nk=20; NbStart=65; NbEnd=104}
)

function Write-InputFile($path, $case) {
  $content = @"
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "$tbFile"
  wannier_hr_file = ""
  wannier_r_file  = ""
/

&kgrid
  nkx = $($case.Nk)
  nky = $($case.Nk)
/

&bands
  nv_orig = 84
  nb_start = $($case.NbStart)
  nb_end = $($case.NbEnd)
/

&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = 2.0e11
  theta_deg = 0.0
  phi_cep_deg = 90.0
  ncyc = 4.0
  env_type = 2
  ellipticity = 0.0
  delta_phase_deg = 90.0
/

&laser2
  wvl_nm_2 = 0.0
  intensity_Wcm2_2 = 0.0
/

&external_field
  use_external_A = .false.
  external_A_file = ""
/

&timestep
  dt = 0.35
  n_dt_deph = 5
/

&dephasing
  T2_fs = 1.0e30
/

&bsv
  bsv_enabled = .false.
  bsv_n_samples = 100
  bsv_mean_intensity = 2.0e11
  bsv_seed = 42
/

&method
  gauge_method = 'vg'
/

&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .true.
/
"@
  Set-Content -Path $path -Value $content -Encoding ASCII
}

$env:OMP_NUM_THREADS = "4"
$env:OMP_STACKSIZE = "256M"

$summary = Join-Path $root "run_summary.csv"
"case,nk,nb_start,nb_end,exit_code,elapsed_seconds" | Set-Content -Path $summary -Encoding ASCII

foreach ($case in $cases) {
  $caseDir = Join-Path $root $case.Name
  New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
  $input = Join-Path $caseDir "input.nml"
  Write-InputFile $input $case

  Write-Host "=== RUN $($case.Name) ==="
  Push-Location $caseDir
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $exe input.nml *> run.log
  $exit = $LASTEXITCODE
  $sw.Stop()
  Pop-Location

  "$($case.Name),$($case.Nk),$($case.NbStart),$($case.NbEnd),$exit,$([math]::Round($sw.Elapsed.TotalSeconds,3))" |
    Add-Content -Path $summary -Encoding ASCII
  if ($exit -ne 0) {
    throw "Case $($case.Name) failed with exit code $exit; see $caseDir/run.log"
  }
}

Write-Host "All cases finished: $root"
