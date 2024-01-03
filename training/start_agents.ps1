$agent_count = 15

$processes = @()
try
{
    $processes += start-process -FilePath python -ArgumentList "agent_gradient.py" -NoNewWindow -PassThru

    Write-Output "Waiting for file to exist"
    While (!(Test-Path -Path .sweep_id -ErrorAction SilentlyContinue))
    {
        # endless loop, when the file will be there, it will continue
    }

    Write-Output "File exists, starting training"
    for ($i = 0; $i -lt $agent_count - 1; $i++) {
        $processes += start-process -FilePath python -ArgumentList "agent_gradient.py" -NoNewWindow -PassThru
    }

    While ($True)
    {
        $running_processes = $processes | Where-Object { $_.HasExited -eq $false }
        if ($running_processes.Count -eq 0)
        {
            break
        }
        else
        {
            Write-Output "Still running: $( $running_processes.Count )"
            Start-Sleep -Seconds 10
        }
    }
}
finally
{
    for($i = 0; $i -lt $processes.Count; $i++)
    {
        Write-Host "Killing process: $( $processes[$i].Id )"
        Stop-Process -Id $processes[$i].Id
    }

    Write-Host "goodbye!"
}
