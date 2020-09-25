import os
import sys


output_dir = "ae_output_slim2car"

for i in range(int(sys.argv[1])):
    idx = i
    depth = 7
    offset = 0.55
    points_in = "%s/%s_input.points" % (output_dir, str(idx).zfill(5))
    mesh_out = "%s/%s_output.obj" % (output_dir, str(idx).zfill(5))

    os.system("~/dev/implicit_ocnn/build/points2obj %s %s_in.obj %d %f" % (points_in, idx, depth, offset))
    os.system("cp %s %s_out.obj" % (mesh_out, idx))