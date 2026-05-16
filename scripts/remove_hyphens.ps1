$root = "c:\Users\yasiru\Desktop\fyp-3d-ct\documents\Progress Report"
Get-ChildItem -Path $root -Recurse -Include *.tex,*.bib,*.sty,*.cfg | ForEach-Object {
    $p = $_.FullName
    try {
        $t = Get-Content -Raw -Encoding UTF8 $p
    } catch {
        $t = Get-Content -Raw $p
    }
    $nt = $t -replace '\-',''
    if ($t -ne $nt) {
        Set-Content -Encoding UTF8 -Value $nt -Path $p
        Write-Output "Edited: $p"
    }
}