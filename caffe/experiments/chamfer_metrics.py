import os
import numpy as np

# executables
caffe = "~/dev/caffe-official/build/install/bin/caffe"
octree_builder = "~/dev/O-CNN/octree/build/octree"
octree2mesh = "~/dev/O-CNN/octree/build/octree2mesh"
octree2points = "~/dev/O-CNN/octree/build/octree2points"
mesh2points = "~/dev/O-CNN/octree/build/mesh2points"
normalize_points = "~/dev/implicit_ocnn/build/normalize_points"
points2obj = "~/dev/implicit_ocnn/build/points2obj"
chamfer_dist = "~/dev/O-CNN/octree/build/chamfer_distance"

# input 
dataset = "ae74_slim2_car"
points_dir = "/media/jeri/DATA/dev/datasets/ShapeNetCore.v2/1_points/baseline/"
octree_dir = "/media/jeri/DATA/dev/datasets/ShapeNetCore.v2/2_octree/%s/" % dataset
filelist_input = "/media/jeri/DATA/dev/datasets/ShapeNetCore.v2/3_lmdb/%s/oct_test_shuffle.txt" %dataset

# auto encoder
depth = 7
offset = 0.55
model = "ae_7_4.test.prototxt"
weights = "ae74.slim2.car.caffemodel"
ae_out_dir = "ae_output_slim2car/"
output_dir = os.path.join(os.getcwd(), ae_out_dir)


num_in = 0


###             INPUT POINT CLOUDS
# read auto encoder input list
files_input = open(filelist_input, 'r').read().split('\n')
categories = [int(f.split(" ")[-1]) for f in files_input[:-1]]
files_octree_in = [os.path.join(octree_dir, f) for f in files_input]

files_octree_in = [os.path.join(octree_dir, f[:f.index("000.octree")+10]) for f in files_input if "000.octree" in f]
if num_in > 0:
    files_octree_in = files_octree_in[:num_in]


files_input = [f[:f.index("000.octree")-5] for f in files_input if "000.octree" in f]
files_input = [os.path.join(points_dir, f.replace("octree", "points") + ".points") for f in files_input]
if num_in > 0:
    files_input = files_input[:num_in]

# copy and normalize input point cloud so chamfer distance is calculated properly
files_input_norm = [os.path.join(output_dir, "%s_input.points" % str(i).zfill(5)) for i in range(len(files_input))]
for i, f in enumerate(files_input):
    cmd = "%s %s %s %d %f" % (normalize_points, f, files_input_norm[i], depth, offset)
    os.system(cmd)

# generate input points list file for chamfer distance 
file_list_points_in = os.path.join(os.getcwd(), "list_points_in.txt")
with open(file_list_points_in, 'w') as flist:
    flist.write("\n".join(files_input_norm))





###             AUTO ENCODER
# generate auto encoder output
num_iters = len(files_input)
cmd = "%s test --model=%s --weights=%s --blob_prefix=%s --blob_header=false --iterations=%d" % (caffe, model, weights, ae_out_dir, num_iters)
os.system(cmd)





###             OUTPUT 
def generate_output_file_list(file_list, suffix, num):
    path_list = os.path.join(os.getcwd(), file_list)
    file_names = [os.path.join(output_dir, "%s_output.%s" % (str(i).zfill(5), suffix)) for i in range(num)]
    with open(file_list, 'w') as flist:
        flist.write("\n".join(file_names))
    return path_list, file_names

# generate output file list for all file types
file_list_octree, files_octree = generate_output_file_list("list_octree.txt", "octree", num_iters)
generate_output_file_list("list_mesh.txt", "obj", num_iters)
file_list_points_out, files_points = generate_output_file_list("list_points_out.txt", "points", num_iters)

# convert octree2mesh -> mesh2points
if True:
    # convert octree2mesh
    cmd = "%s --filenames %s --output_path %s --pu 0 --depth_start 0" % (octree2mesh, file_list_octree, ae_out_dir)
    print(cmd)
    os.system(cmd)

    # convert mesh2points
    cmd = "%s --filenames %s --output_path %s --area_unit 1.0" % (mesh2points, os.path.join(os.getcwd(), "list_mesh.txt"), ae_out_dir)
    print(cmd)
    os.system(cmd)

# too few samples to calculate chamfer distance properly
# convert octrees2points directly
else:
    cmd = "%s --filenames %s --output_path %s --depth_end %d" % (octree2points, file_list_octree, ae_out_dir, depth)
    print(cmd)
    os.system(cmd)




###             CHAMFER DISTANCE 

# calc chamfer distance
os.system("rm list_chamfer.txt")
cmd = "%s --filename_out %s --path_a %s --path_b %s" % (chamfer_dist, "list_chamfer.txt", file_list_points_in, file_list_points_out)
print(cmd)
os.system(cmd)


# calculate average per category distance
metric = ["category, avg_ab, avg_ba, avg_ab + avg_ba"]

num_categories = len(set(categories))
categories = np.array(categories)
distances = open("list_chamfer.txt", 'r').read().split("\n")[:-1]
distances = np.array([[float(val.replace(",", "")) for val in l.split(" ")[-3:]] for l in distances])

for i in range(num_categories):
    idx = (categories == i)[:distances.shape[0]]
    m = np.mean(distances[idx], axis=0)
    metric.append("%d, %f, %f, %f" % (i, m[0], m[1], m[2])) 

with open("list_chamfer_category.txt", 'w') as fmetric:
    fmetric.write("\n".join(metric))






###             DEBUG

# convert points to obj point cloud to debug scale issue
if False:
    mid = 3 # index of output mesh

    # point cloud to obj
    for i, f in enumerate([files_input_norm[mid], files_points[mid]]):
        cmd = "%s %s %s %d %f 0" % (points2obj, f, "%d.obj" % i, depth, offset)
        os.system(cmd)

    # input point cloud to octree
    with open(file_list_octree, 'w') as flist:
        flist.write(files_input[mid])
        print(files_input[mid])
    cmd = "%s --filenames %s --output_path %s --depth %d --adaptive 1 --node_dis 1 --axis y --rot_num 1 --offset %f" % (octree_builder, file_list_octree, os.getcwd(), depth, offset)
    print(cmd)
    os.system(cmd)



    # output octree to obj
    with open(file_list_octree, 'w') as flist:
        flist.write(files_octree[mid] + "\n")
        flist.write(files_octree_in[mid] + "\n")
        flist.write(os.path.join(os.getcwd(), "%s_input_7_2_000.octree" % str(mid).zfill(5)))
    cmd = "%s --filenames %s --output_path %s --pu 0 --depth_start 0" % (octree2mesh, file_list_octree, os.getcwd())
    os.system(cmd)


