# M5 handoff setup (Windows / PowerShell)

This source-and-results package intentionally excludes the raw dataset, extracted images, and large `.pt` checkpoint binaries. They are supplied separately in the two uploaded artifact archives.

## 1. Merge the source package
Extract `AI403_M5_Source_and_Results.zip`. Open the resulting `AI403-Handwritten-Digit-Recognition-M5` folder in VS Code.

## 2. Place the ORAND archive
Create `data\raw` and copy the original archive there as:
`data\raw\ORAND-CAR-2014.tar.gz`

Verify the SHA-256:
`Get-FileHash .\data\raw\ORAND-CAR-2014.tar.gz -Algorithm SHA256`

Expected:
`7c4b3d9e820bbcdfd291f5679b74013bd055b4f771dc64ebd8b8ce599ff4a61a`

Then run:
`python data/prepare_images.py`

This extracts only the frozen TRAIN/VALIDATION images. It does not open the official test labels.

## 3. Merge model checkpoints
Create `checkpoints` at the project root. Extract the inner ZIPs from `Checkpoints_Baseline_Classic_ResNet18_Scratch_Classic_ResNet18_Pretrained_Modern-...zip` into the project root (not inside another nested folder). Each inner archive creates its own `checkpoints\<model>_results\...` tree:
- `baseline_results.zip`
- `classic_scratch_results.zip`
- `classic_pretrained_results.zip`
- `modern_results.zip`

For M4, open `AI403_M4_Advanced_Artifacts-...zip` and extract these two folders from `AI403_M4_Advanced_Artifacts\checkpoints\` into the project's `checkpoints\` folder:
- `advanced_results`
- `advanced_no_multiscale_results`

The final project needs final seeds 7, 21, and 42 for each included model family. Do not copy `seed_7_interrupted`.

## 4. Install and run
From the project root:
```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_m5_integration.py" -v
python -m streamlit run demo/app.py
```

To regenerate the comparison CSV and charts:
```powershell
python m5/build_comparison.py
```

## 5. Important evidence notes
- M5 integration tests verify checkpoint presence, strict state loading, preprocessing, and a forward prediction for each bundled model family.
- The supplied M4 official-test summary is preserved as received. Its provenance document has unfilled fields; see `docs/M4_Official_Test_Evidence_Package/`.
- The `advanced_no_attention` architecture is registered in the source but the supplied M4 contribution log says matched three-seed training/evaluation was not completed. It is not presented as a trained final model in the demo.
