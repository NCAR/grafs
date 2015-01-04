import subprocess
import os
from datetime import datetime, timedelta
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start",help="Initial date")
    parser.add_argument("end",help="Final date")
    parser.add_argument("--file",default="grid_int_fcst",help="File beginning")
    parser.add_argument("--hpath",default="/MDSS/DICAST/minivet/nt/",help="Path to files on HPSS")
    parser.add_argument("--out",default="/d2/dgagne/grid_int_fcst/",help="Output path")
    args = parser.parse_args()
    date_format = "%Y%m%d"
    start_date = datetime.strptime(args.start, date_format)
    end_date = datetime.strptime(args.end, date_format)
    curr_date = start_date
    while curr_date <= end_date:
        print curr_date
        get_forecast_file(curr_date, args.file, args.hpath, args.out)
        curr_date += timedelta(days=1)
    return

def get_forecast_file(date_obj, file_type, in_path, out_path):
    date_str = date_obj.strftime("%Y%m%d")
    os.chdir(out_path)
    filename = "{0}{1}/{2}.{1}.tar.gz".format(in_path,date_str,file_type)
    cmd = "hsi -P -q get {0}".format(filename)
    print cmd
    subprocess.call(cmd,shell=True)
    return

if __name__ == "__main__":
    main()

