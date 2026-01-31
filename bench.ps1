$fullArgs = $args

if ($fullArgs.Count -eq 0) {
    Write-Host "Usage: .\bench.ps1 <votre_executable.exe>" -ForegroundColor Yellow
    exit 1
}

$trace  = "cuda"
$output = "nsys_easy"
$report = "cuda_gpu_sum"

Write-Host "--- Profiling $($fullArgs -join ' ') ---" -ForegroundColor Green

& nsys profile --trace=$trace --sample=none --cpuctxsw=none --force-overwrite=true -o $output -- $fullArgs

Write-Host "`n--- Report: $report ---" -ForegroundColor Green
& nsys stats --force-export=true -r $report "$output.nsys-rep"