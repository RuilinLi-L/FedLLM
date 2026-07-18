param(
    [string]$RunsRoot = "log/runs",
    [string]$OutputPath = "docs/DAGER_PRIVACY_ALL_PARAMETERS_NO_OLD_20260715.md",
    [switch]$IncludeOld,
    [switch]$EmitStdout,
    [ValidateSet('full', 'header', 'baseline', 'attempts', 'footer')]
    [string]$EmitPart = 'full',
    [string]$BaselineFilter = ''
)

$ErrorActionPreference = "Stop"

function Get-RelativePath([string]$BasePath, [string]$Path) {
    $base = [IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/')
    $full = [IO.Path]::GetFullPath($Path)
    return $full.Substring($base.Length).TrimStart('\', '/')
}

function Get-SourceRoot([string]$RunsPath, [string]$Path) {
    $relative = Get-RelativePath $RunsPath $Path
    return ($relative -split '[\\/]')[0]
}

function Get-Seed([object]$Row) {
    if ($null -ne $Row.seed -and -not [string]::IsNullOrWhiteSpace([string]$Row.seed)) {
        return [string]$Row.seed
    }
    $name = [IO.Path]::GetFileName([string]$Row.log_path)
    $match = [regex]::Match($name, '(?i)seed[_-]?(\d+)')
    if ($match.Success) {
        return $match.Groups[1].Value
    }
    return ""
}

function Get-Protocol([object]$Row) {
    $partial = [string]$Row.partial_attack_variant
    if ($partial -eq 'ptg_gradient_matching') {
        return 'exclude-ptg'
    }

    $trainMethod = [string]$Row.train_method
    $peftMethod = [string]$Row.peft_method
    if ($trainMethod -eq 'peft' -or ($peftMethod -and $peftMethod -notin @('n/a', 'none'))) {
        if ($peftMethod -and $peftMethod -notin @('n/a', 'none')) {
            return "PEFT DAGER ($peftMethod)"
        }
        return 'PEFT DAGER'
    }

    $adaptive = [string]$Row.adaptive_attack
    if ($adaptive -and $adaptive -notin @('none', 'false', 'n/a')) {
        return "adaptive full-gradient DAGER ($adaptive)"
    }

    $partialActive = [string]$Row.partial_filter_active
    if ($partialActive -eq 'true' -or ($partial -and $partial -notin @('n/a', 'none', 'full_gradient_visible'))) {
        $subset = [string]$Row.gradient_layer_subset
        if (-not $subset -or $subset -eq 'n/a') {
            $subset = [string]$Row.selected_block_ids
        }
        if (-not $subset -or $subset -eq 'n/a') {
            $subset = 'unspecified'
        }
        if (-not $partial -or $partial -eq 'n/a') {
            $partial = 'partial'
        }
        return "partial-gradient DAGER ($partial; $subset)"
    }
    return 'standard full-gradient DAGER'
}

function Convert-ToDouble([object]$Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value) -or [string]$Value -eq 'n/a') {
        return $null
    }
    try {
        return [double]::Parse([string]$Value, [Globalization.CultureInfo]::InvariantCulture)
    }
    catch {
        return $null
    }
}

function Format-Stats([object[]]$Values) {
    $numbers = @($Values | ForEach-Object { Convert-ToDouble $_ } | Where-Object { $null -ne $_ })
    if ($numbers.Count -eq 0) {
        return '--'
    }
    if ($numbers.Count -eq 1) {
        return ('{0:F6}' -f $numbers[0])
    }
    $mean = ($numbers | Measure-Object -Average).Average
    $sumSq = 0.0
    foreach ($number in $numbers) {
        $sumSq += [Math]::Pow($number - $mean, 2)
    }
    $std = [Math]::Sqrt($sumSq / $numbers.Count)
    return ('{0:F6} +/- {1:F6}' -f $mean, $std)
}

function Escape-Markdown([object]$Value) {
    if ($null -eq $Value) {
        return ''
    }
    return ([string]$Value).Replace('|', '\|').Replace("`r", ' ').Replace("`n", ' ')
}

$runsFull = [IO.Path]::GetFullPath((Join-Path (Get-Location) $RunsRoot))
$outputFull = [IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputPath))

$resultFiles = @(Get-ChildItem -Recurse -File -LiteralPath $runsFull -Filter 'results.csv')
if (-not $IncludeOld) {
    $resultFiles = @($resultFiles | Where-Object { (Get-SourceRoot $runsFull $_.FullName) -ne 'old' })
}

$rawRecords = New-Object System.Collections.Generic.List[object]
foreach ($file in $resultFiles) {
    $sourceRoot = Get-SourceRoot $runsFull $file.FullName
    $sourceUnit = Get-RelativePath $runsFull $file.Directory.FullName
    foreach ($row in (Import-Csv -LiteralPath $file.FullName)) {
        if ([string]$row.log_kind -ne 'attack_dager') {
            continue
        }
        $protocol = Get-Protocol $row
        if ($protocol -eq 'exclude-ptg') {
            continue
        }

        $parameter = [string]$row.defense_param_value
        if (-not $parameter) {
            $parameter = 'n/a'
        }
        $checkpoint = [string]$row.finetuned_path
        if (-not $checkpoint -or $checkpoint -eq 'n/a') {
            $checkpoint = [string]$row.model_path_guess
        }
        if (-not $checkpoint) {
            $checkpoint = 'n/a'
        }
        $logPath = [string]$row.log_path
        if (-not $logPath) {
            $logPath = "$($file.FullName)#$($rawRecords.Count)"
        }

        $rawRecords.Add([pscustomobject]@{
            LogKey = $logPath
            LogPath = [string]$row.log_path
            SourceCsv = $file.FullName
            SourceRoot = $sourceRoot
            SourceUnit = $sourceUnit
            Protocol = $protocol
            Dataset = [string]$row.dataset
            BatchSize = [string]$row.batch_size
            NInputsRequested = [string]$row.n_inputs_requested
            NInputsCompleted = [string]$row.n_inputs_completed
            TrainMethod = [string]$row.train_method
            PeftMethod = [string]$row.peft_method
            PartialVariant = [string]$row.partial_attack_variant
            GradientSubset = [string]$row.gradient_layer_subset
            AdaptiveProfile = [string]$row.adaptive_attack_profile
            Checkpoint = $checkpoint
            Defense = [string]$row.defense
            ParameterName = [string]$row.defense_param_name
            Parameter = $parameter
            Seed = Get-Seed $row
            ResultStatus = [string]$row.result_status
            RecToken = [string]$row.rec_token_mean
            Rouge = [string]$row.agg_r1fm_r2fm
        })
    }
}

$deduplicated = foreach ($group in ($rawRecords | Group-Object LogKey)) {
    $group.Group |
        Sort-Object @{ Expression = { if ($_.ResultStatus -eq 'ok') { 1 } else { 0 } }; Descending = $true },
                    @{ Expression = { $value = Convert-ToDouble $_.NInputsCompleted; if ($null -eq $value) { -1 } else { $value } }; Descending = $true } |
        Select-Object -First 1
}

$groupedRows = New-Object System.Collections.Generic.List[object]
$groupKey = {
    @(
        $_.SourceUnit, $_.Protocol, $_.Dataset, $_.BatchSize, $_.NInputsRequested,
        $_.TrainMethod, $_.PeftMethod, $_.PartialVariant, $_.GradientSubset,
        $_.AdaptiveProfile, $_.Checkpoint, $_.Defense, $_.ParameterName, $_.Parameter
    ) -join [char]31
}

foreach ($group in ($deduplicated | Group-Object $groupKey)) {
    $first = $group.Group[0]
    $valid = @($group.Group | Where-Object {
        $requested = Convert-ToDouble $_.NInputsRequested
        $completed = Convert-ToDouble $_.NInputsCompleted
        $_.ResultStatus -eq 'ok' -and $null -ne $requested -and $requested -gt 0 -and $completed -eq $requested
    })
    $failed = @($group.Group | Where-Object { $_.ResultStatus -eq 'failed' })
    $other = $group.Group.Count - $valid.Count - $failed.Count

    if ($valid.Count -gt 0) {
        $seeds = @($valid.Seed | Where-Object { $_ } | Sort-Object -Unique)
        if ($valid.Count -eq 1) {
            if ($seeds.Count -eq 1) {
                $seedStatus = "seed $($seeds[0]) (single seed)"
            }
            else {
                $seedStatus = 'single run (seed not recorded)'
            }
        }
        elseif ($seeds.Count -gt 0) {
            $seedStatus = "$(($seeds -join '/')) ($($seeds.Count) seeds)"
        }
        else {
            $seedStatus = "$($valid.Count) runs (seeds not recorded)"
        }
        if ($failed.Count -gt 0 -or $other -gt 0) {
            $seedStatus += "; failed=$($failed.Count), other=$other"
        }
        if ($seeds.Count -gt 1) {
            $rec = Format-Stats $valid.RecToken
            $rouge = Format-Stats $valid.Rouge
        }
        elseif ($valid.Count -eq 1) {
            $rec = Format-Stats $valid.RecToken
            $rouge = Format-Stats $valid.Rouge
        }
        else {
            $recValues = @($valid.RecToken | ForEach-Object { Convert-ToDouble $_ } | Where-Object { $null -ne $_ } | ForEach-Object { '{0:F6}' -f $_ } | Sort-Object -Unique)
            $rougeValues = @($valid.Rouge | ForEach-Object { Convert-ToDouble $_ } | Where-Object { $null -ne $_ } | ForEach-Object { '{0:F6}' -f $_ } | Sort-Object -Unique)
            $rec = if ($recValues.Count -gt 0) { $recValues -join '/' } else { '--' }
            $rouge = if ($rougeValues.Count -gt 0) { $rougeValues -join '/' } else { '--' }
        }
    }
    else {
        $completion = @($group.Group.NInputsCompleted | Where-Object { $_ } | Sort-Object -Unique) -join '/'
        if (-not $completion) {
            $completion = 'n/a'
        }
        $seedStatus = "no valid result; failed=$($failed.Count), other=$other; completed=$completion"
        $rec = '--'
        $rouge = '--'
    }

    $groupedRows.Add([pscustomobject]@{
        Defense = $first.Defense
        Protocol = $first.Protocol
        Dataset = $first.Dataset
        BatchSize = $first.BatchSize
        NInputs = $first.NInputsRequested
        Parameter = $first.Parameter
        RecToken = $rec
        Rouge = $rouge
        SeedStatus = $seedStatus
        SourceRoot = $first.SourceUnit
    })
}

$attemptOnly = New-Object System.Collections.Generic.List[object]
$headers = @(Get-ChildItem -Recurse -File -LiteralPath $runsFull -Filter '_run_header.txt')
foreach ($header in $headers) {
    $sourceRoot = Get-SourceRoot $runsFull $header.FullName
    if (-not $IncludeOld -and $sourceRoot -eq 'old') {
        continue
    }
    $runDir = $header.Directory.FullName
    if (Test-Path -LiteralPath (Join-Path $runDir 'results.csv')) {
        continue
    }
    $text = Get-Content -Raw -LiteralPath $header.FullName
    $firstLine = ($text -split "`r?`n")[0]
    $argvMatch = [regex]::Match($firstLine, 'argv:\s+(?<argv>.+?)\s+=====$')
    $argv = if ($argvMatch.Success) { $argvMatch.Groups['argv'].Value } else { '' }
    $tokens = @($argv -split '\s+')
    $dataset = if ($tokens.Count -ge 1) { $tokens[0] } else { 'unknown' }
    $batch = if ($tokens.Count -ge 2) { $tokens[1] } else { 'unknown' }
    $nInputs = if ($tokens.Count -ge 4) { $tokens[3] } else { 'unknown' }
    $defenseMatch = [regex]::Match($text, '(?m)^focus_baseline_defense=(.+)$')
    $paramMatch = [regex]::Match($text, '(?m)^focus_baseline_param=(.+)$')
    $seedMatch = [regex]::Match($text, '(?m)^seeds=(.+)$')
    $defense = if ($defenseMatch.Success) { $defenseMatch.Groups[1].Value.Trim() } else { 'all/default' }
    $parameter = if ($paramMatch.Success) { $paramMatch.Groups[1].Value.Trim() } else { 'all/default' }
    $seeds = if ($seedMatch.Success) { $seedMatch.Groups[1].Value.Trim() } else { 'not recorded' }
    $protocol = if ($runDir -match 'partial_gradient') { 'partial-gradient DAGER attempt' } else { 'standard full-gradient DAGER attempt' }

    $attemptOnly.Add([pscustomobject]@{
        Protocol = $protocol
        Dataset = $dataset
        BatchSize = $batch
        NInputs = $nInputs
        Defense = $defense
        Parameter = $parameter
        Seeds = $seeds
        SourceRoot = $sourceRoot
        RelativePath = Get-RelativePath $runsFull $runDir
    })
}

$baselineOrder = @('none', 'lrbprojonly', 'lrb', 'topk', 'compression', 'noise', 'dpsgd', 'dpsgd_opacus', 'soteria', 'mixup')
$baselineNames = @{
    none = 'none'
    lrbprojonly = 'Projection-LRB / lrbprojonly'
    lrb = 'LRB variants / Full-LRB'
    topk = 'top-k'
    compression = 'compression'
    noise = 'noise'
    dpsgd = 'DP-SGD-style'
    dpsgd_opacus = 'Opacus DP-SGD'
    soteria = 'Soteria'
    mixup = 'Mixup'
}

$builder = New-Object Text.StringBuilder
if ($EmitPart -in @('full', 'header')) {
    [void]$builder.AppendLine('# DAGER privacy experiments: all attempted parameters (old excluded)')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('Updated: 2026-07-15')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('Scope: recursively scan `log/runs/**/results.csv`, explicitly exclude `log/runs/old/**`, retain DAGER privacy rows, and exclude PTG rows with `partial_attack_variant=ptg_gradient_matching`. Standard, adaptive, partial-gradient, and PEFT DAGER are labeled separately.')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('Statistics: include numeric metrics only when `result_status=ok` and `n_inputs_completed=n_inputs_requested`. Compute the population standard deviation (`ddof=0`) across seeds only within the same source run, protocol, dataset, batch, baseline, and parameter. A single record is reported without a synthetic standard deviation. Duplicate copies with the same `log_path` are counted once.')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("Inventory: $($resultFiles.Count) results.csv files, $($rawRecords.Count) non-PTG DAGER rows, $($deduplicated.Count) rows after log_path deduplication, $($groupedRows.Count) grouped parameter/run rows, and $($attemptOnly.Count) attempt-only directories without results.csv.")
}

if ($EmitPart -in @('full', 'baseline')) {
foreach ($baseline in $baselineOrder) {
    if ($BaselineFilter -and $baseline -ne $BaselineFilter) {
        continue
    }
    $rows = @($groupedRows | Where-Object { $_.Defense -eq $baseline } | Sort-Object Protocol, Dataset, BatchSize, NInputs, Parameter, SourceRoot)
    if ($rows.Count -eq 0) {
        continue
    }
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("## $($baselineNames[$baseline])")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('| Protocol | Dataset | Batch | n_inputs | Parameter | rec_token_mean | R1+R2 | Seeds/status | Source run |')
    [void]$builder.AppendLine('|---|---|---:|---:|---|---:|---:|---|---|')
    foreach ($row in $rows) {
        $line = '| ' + (Escape-Markdown $row.Protocol) +
            ' | ' + (Escape-Markdown $row.Dataset) +
            ' | ' + (Escape-Markdown $row.BatchSize) +
            ' | ' + (Escape-Markdown $row.NInputs) +
            ' | `' + (Escape-Markdown $row.Parameter) +
            '` | `' + (Escape-Markdown $row.RecToken) +
            '` | `' + (Escape-Markdown $row.Rouge) +
            '` | ' + (Escape-Markdown $row.SeedStatus) +
            ' | `' + (Escape-Markdown $row.SourceRoot) + '` |'
        [void]$builder.AppendLine($line)
    }
}
}

if ($EmitPart -in @('full', 'attempts') -and $attemptOnly.Count -gt 0) {
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('## Attempt-only directories without result files')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('These directories show that a task was created or started, but they contain no `results.csv`; no privacy metric is imputed.')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('| Protocol | Dataset | Batch | n_inputs | Baseline | Parameter scope | Planned seeds | Source run |')
    [void]$builder.AppendLine('|---|---|---:|---:|---|---|---|---|')
    foreach ($row in ($attemptOnly | Sort-Object Defense, Dataset, Parameter, SourceRoot)) {
        $line = '| ' + (Escape-Markdown $row.Protocol) +
            ' | ' + (Escape-Markdown $row.Dataset) +
            ' | ' + (Escape-Markdown $row.BatchSize) +
            ' | ' + (Escape-Markdown $row.NInputs) +
            ' | `' + (Escape-Markdown $row.Defense) +
            '` | `' + (Escape-Markdown $row.Parameter) +
            '` | ' + (Escape-Markdown $row.Seeds) +
            ' | `' + (Escape-Markdown $row.RelativePath) + '` |'
        [void]$builder.AppendLine($line)
    }
}

if ($EmitPart -in @('full', 'footer')) {
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('## Interpretation boundary')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('- The same parameter remains on separate rows when source run, batch, checkpoint, or attack protocol differs; values are not merged across protocols.')
    [void]$builder.AppendLine('- `rec_token_mean=0` means no token was recovered under the current attack setting and budget; it is not a formal privacy guarantee.')
    [void]$builder.AppendLine('- DP-SGD-style rows without privacy accounting do not establish epsilon/delta DP. Failed or incomplete rows are never replaced with zero.')
}

$outputDir = Split-Path -Parent $outputFull
if (-not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$utf8 = New-Object Text.UTF8Encoding($false)
if ($EmitStdout) {
    $reportText = $builder.ToString()
    Write-Output '<<<DAGER_REPORT_START>>>'
    Write-Output $reportText
    Write-Output "<<<DAGER_REPORT_META chars=$($reportText.Length)>>>"
    Write-Output '<<<DAGER_REPORT_END>>>'
    return
}

[IO.File]::WriteAllText($outputFull, $builder.ToString(), $utf8)
Write-Output "Wrote $($groupedRows.Count) grouped parameter rows and $($attemptOnly.Count) attempt-only rows to $outputFull"
