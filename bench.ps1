# On récupère tous les arguments passés au script
$fullArgs = $args

if ($fullArgs.Count -eq 0) {
    Write-Host "Usage: .\bench.ps1 <votre_executable.exe>" -ForegroundColor Yellow
    exit 1
}

# Paramètres par défaut (modifiables ici directement)
$trace  = "cuda"
$output = "nsys_easy"
$report = "cuda_gpu_sum"

Write-Host "--- Profilage de : $($fullArgs -join ' ') ---" -ForegroundColor Cyan

# 1. Exécution du profilage
# -- indique à nsys que tout ce qui suit est l'application cible
& nsys profile --trace=$trace --sample=none --cpuctxsw=none --force-overwrite=true -o $output -- $fullArgs

# 2. Génération des statistiques
Write-Host "`n--- Rapport : $report ---" -ForegroundColor Cyan
& nsys stats --force-export=true -r $report "$output.nsys-rep"