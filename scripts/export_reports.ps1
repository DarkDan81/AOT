param([string[]]$Folders=@('lab01_representations','lab02_annotation','lab03_classification','lab04_prompt_program','lab05_mini_rag','lab06_ai_detective'))
foreach($folder in $Folders) {
 $wordApp=New-Object -ComObject Word.Application
 $wordApp.Visible=$false
 $wordApp.DisplayAlerts=0
 $wordApp.AutomationSecurity=3
 $docPath=[string](Join-Path (Get-Location) ($folder+'/report.docx'))
 $pdfPath=[string](Join-Path (Get-Location) ($folder+'/report.pdf'))
 $reportDoc=$wordApp.Documents.OpenNoRepairDialog($docPath,$false,$false)
 Write-Output ('Opened '+$docPath)
 if($reportDoc.TablesOfContents.Count -gt 0){$reportDoc.TablesOfContents.Item(1).Update()}
 $reportDoc.Save()
 $reportDoc.SaveAs2($pdfPath,17)
 $reportDoc.Close(0)
 $wordApp.Quit()
 Write-Output ('Saved '+$pdfPath)
}
