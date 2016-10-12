import subprocess
from glob import glob
import os
import sys


def main():
    path = sys.argv[1]
    tar_files = sorted(glob(path + "*.tar.gz"))
    for tar_file in tar_files:
        date = tar_file.split(".")[1]
        if not os.access(path + date,os.R_OK):
            print(tar_file)
            os.mkdir(path + date)
            os.chdir(path + date)
            cmd = "tar -xvzf {0}".format(tar_file)
            subprocess.call(cmd, shell=True)
    return

if __name__ == "__main__":
    main()
