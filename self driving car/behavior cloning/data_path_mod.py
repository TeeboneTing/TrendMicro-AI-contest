# modify data path in the log file

import sys

log_file = sys.argv[1]
current_img_path = sys.argv[2]

with open(log_file,"r") as f:
    for line in f:
        split = line.split(',')
        path, others = split[0], split[1:]
        new_path = current_img_path + "/" + path.split("\\")[-1]
        print(new_path + "," + ",".join(others).strip())
