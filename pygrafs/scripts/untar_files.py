import subprocess
from glob import glob
from os.path import exists, join
import os
import sys


def main():
    path = sys.argv[1]
    tar_files = sorted(glob(path + "*.tar.gz"))
    for tar_file in tar_files:
        date = tar_file.split(".")[1]
        if not exists(join(path, date)):
            print(tar_file)
            os.mkdir(join(path, date))
            os.chdir(join(path, date))
            cmd = "tar -xvzf {0} {1}".format(tar_file, "def_fcst.{0}.00.nc".format(date))
            subprocess.call(cmd, shell=True)
    return

if __name__ == "__main__":
    main()
