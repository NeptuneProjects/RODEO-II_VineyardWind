python scripts/setup/acoustic_inventory.py
python scripts/process/strikes_find.py
python scripts/process/strikes_save.py
python scripts/process/strikes_corr.py --max-workers 256
python scripts/process/template_extraction.py --save-plots
