echo "build"
sh scripts/build.sh
echo "export"
sh scripts/export.sh
echo "load"
sh scripts/load.sh
echo "run"
python run.py
