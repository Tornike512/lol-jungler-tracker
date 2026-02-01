  cd tlol                                                                                                                           python setup.py
                                                                                                                                  
  This will download the dataset fresh from Google Drive (takes ~1 minute).

  Then extract it:
  python -c "import py7zr; py7zr.SevenZipFile('data/ezreal_converted.zip').extractall('data/ezreal_data')"

  Then train:
  python train_ezreal.py --max_games 5000 --epochs 50

  ---
  What to push to git (just the code):
  - setup.py - downloads dataset
  - train_ezreal.py - training script
  - explore_data.py - data exploration
  - .gitignore - excludes data files

  The data gets downloaded fresh on each machine. Much faster than pushing through git.

● Ran 1 stop hook