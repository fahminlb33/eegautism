import os
import glob
import subprocess

files = glob.glob("data/bci2000/normal/*.*")
for file in files:
    output_name = os.path.splitext(os.path.basename(file))[0] + ".edf"
    output_path = os.path.join("data/edf/normal", output_name)
    subprocess.call(["bin2rec", "-f=EDF", file, output_path])
