<#
Reproducible Applied Ocean Research IL evaluation on Windows (from project root).
No model training occurs; the models and tests are frozen.
Pass -QuickCheck for a 20 s simulator smoke test, NOT a publishable result.
For a full prospective report omit -QuickCheck and budget long CPU time.
#>
param(
  [switch]$QuickCheck,
  [switch]$IncludeTeacher,
  [int]$TaskLimit = 0,
  [int]$HardTimeoutSeconds = 1800,
  [string]$Python = "python",
  [string]$Out = "results/prospective_v1"
)
$ErrorActionPreference = "Stop"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"
& $Python -m pytest -q tests tests_il
if ($LASTEXITCODE -ne 0) { throw "Tests did not pass; investigate before launching experiments." }
& $Python -m research_protocol.runner --dry-run --limit-tasks 2 --seeds 1049
if ($LASTEXITCODE -ne 0) { throw "Protocol preflight failed." }
$methods = @('routed_il', 'baseline_il')
if ($IncludeTeacher) { $methods += 'teacher_nmpc' }
$args = @('-m','research_protocol.runner','--methods') + $methods + @('--out',$Out,'--hard-timeout-s',$HardTimeoutSeconds)
if ($TaskLimit -gt 0) {$args += @('--limit-tasks',$TaskLimit)}
if ($QuickCheck) { $args += @('--limit-tasks','1','--seeds','1049','--duration','20') }
Write-Host "Launching evaluation with frozen parameters. Output: $Out" -ForegroundColor Cyan
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "Experimental runner failed; check logs." }
& $Python -m research_protocol.report --results $Out
if ($LASTEXITCODE -ne 0) { throw "Report generation failed." }
Write-Host "Results saved under $Out" -ForegroundColor Green
