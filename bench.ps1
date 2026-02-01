$fullArgs = $args

if ($fullArgs.Count -eq 0) {
    Write-Host "Usage: .\bench.ps1 <votre_executable.exe> [args...]" -ForegroundColor Yellow
    exit 1
}

$trace  = "cuda"
$output = "nsys_easy"
$report = "cuda_gpu_sum"

Write-Host "--- Profiling $($fullArgs -join ' ') ---" -ForegroundColor Green

# Détection des options supportées par CETTE installation de nsys
$helpText = (& nsys profile --help 2>&1 | Out-String)

$profileArgs = @(
    "profile",
    "--trace=$trace",
    "--sample=none",
    "--cpuctxsw=none",
    "--force-overwrite=true",
    "-o", $output,
    "--"
) + $fullArgs

# Option utile, généralement dispo sur Windows aussi
if ($helpText -match "--cuda-memory-usage") {
    # forme "option valeur" pour éviter les surprises de parsing
    $profileArgs = $profileArgs[0..3] + @("--cuda-memory-usage", "true") + $profileArgs[4..($profileArgs.Count-1)]
}

# Options UM: souvent absentes sur Windows, présentes sur Linux
if ($helpText -match "--cuda-um-cpu-page-faults" -and $helpText -match "--cuda-um-gpu-page-faults") {
    $profileArgs = $profileArgs[0..3] + @(
        "--cuda-um-cpu-page-faults", "true",
        "--cuda-um-gpu-page-faults", "true"
    ) + $profileArgs[4..($profileArgs.Count-1)]
} else {
    Write-Host "Note: UM page faults non supportes par ce nsys (typique sur Windows), options ignorees." -ForegroundColor DarkYellow
}

& nsys @profileArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "nsys profile a echoue (code $LASTEXITCODE). Abandon." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n--- Report: $report ---" -ForegroundColor Green

# Ne pas forcer la re-exportation
& nsys stats -r $report "$output.nsys-rep"
if ($LASTEXITCODE -ne 0) {
    Write-Host "nsys stats a echoue (code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

# Nettoyage: supprimer le SQLite que nsys genere pour certains rapports
$sqliteBase = "$output.sqlite"
Remove-Item $sqliteBase -Force -ErrorAction SilentlyContinue
Remove-Item "$sqliteBase-wal" -Force -ErrorAction SilentlyContinue
Remove-Item "$sqliteBase-shm" -Force -ErrorAction SilentlyContinue
Remove-Item "$sqliteBase-journal" -Force -ErrorAction SilentlyContinue