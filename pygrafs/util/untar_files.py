import subprocess
from glob import glob
import os

def main():
    path = "/d2/dgagne/grid_int_fcst/"
    tar_files = sorted(glob(path + "*.tar.gz"))
    for tar_file in tar_files:
        print tar_file
        date = tar_file.split(".")[1]
        if not os.access(path + date,os.R_OK):
            os.mkdir(path + date)
        os.chdir(path + date)
        cmd = "tar -xvzf {0}".format(tar_file)
        subprocess.call(cmd,shell=True)
    return

if __name__ == "__main__":
    main()
