import os
import subprocess
import numpy

######
process_img = False
convert_ply = False
generate_octree = True

depth = 7

print('Please properly configure the following 5 variables')
root_pc = '/home/jeri/dev/data/ShapeNetCore.v1/1_points_baseline'
root_img = ''
root_lmdb = '/home/jeri/dev/data/ShapeNetCore.v1/1_points_baseline/'
root_octree = '/home/jeri/dev/O-CNN/octree/build'
root_caffe = '/home/jeri/dev/3rdparty/caffe-official/build/install/bin'

######
ply2points = os.path.join(root_octree, 'ply2points')
octree = os.path.join(root_octree, 'octree')
convert_octree = os.path.join(root_caffe, 'convert_octree_data')
convert_image = os.path.join(root_caffe, 'convert_imageset')

category = [
  '02691156',
  '02828884',
  '02933112',
  '02958343',
  '03001627', 
  '03211117',
  '03636649',
  '03691459',
  '04090263',
  '04256520',
  '04379243',
  '04401088',
  '04530566'
]


# generate octree
print('Generate octree ...')

path_datalist = os.path.join(root_pc, 'datalist')
if not os.path.exists(path_datalist):
  os.mkdir(path_datalist)

for i in range(0, len(category)):
  # path
  print('Processing ' + category[i])
  path_category = os.path.join(root_pc, category[i])
  path_ply = os.path.join(path_category, 'ply')
  path_points = os.path.join(path_category, 'points')
  path_octree = os.path.join(path_category, 'octree')
  
  # remove file _.points.ply
  filename = os.path.join(path_ply, '_.points.ply')
  if os.path.exists(filename):
    os.remove(filename)
  
  # generate the datalist for ply2points.exe
  if convert_ply:
    filename_ply = os.listdir(path_ply)
    filename_list = os.path.join(path_datalist, category[i] + '_ply_list.txt')
    with open(filename_list, 'w') as f:
      for item in filename_ply:
        if item.endswith('.ply'):
          f.write('%s/%s\n' % (path_ply, item))
  
    # call ply2points.exe
    # print(
    subprocess.check_call(
      [ply2points, '--filenames', filename_list, '--output_path', path_points]
    )

  # generate the datalist for octree.exe
  if generate_octree:
    filename_points = os.listdir(path_points)

    # filter files out that are already processed
    if os.path.exists(path_octree):
      filename_octree = os.listdir(path_octree)
      filename_octree = [f.split("/")[-1].replace("_%d_2_000.octree" % depth, ".points0") for f in filename_octree]
      filename_octree = [f.split("/")[-1].replace("_%d_2_001.octree" % depth, ".points1") for f in filename_octree]
      filename_octree = [f.split("/")[-1].replace("_%d_2_011.octree" % depth, ".points11") for f in filename_octree]
      filename_points = [f for f in filename_points if f+"0" not in filename_octree or f+"1" not in filename_octree or f+"11" not in filename_octree]

    print(category[i], len(filename_points))

    filename_list = os.path.join(path_datalist, category[i] + '_points_list.txt')
    with open(filename_list, 'w') as f:
      for item in filename_points:
        f.write('%s/%s\n' % (path_points, item))
    
    # call octree.exe
    # print(
    subprocess.check_call(
      [octree, '--filenames', filename_list, '--output_path', path_octree, 
      '--adp_depth', '4', '--depth', str(depth),
      '--adaptive', '1', '--th_distance', '0.7', '--th_normal', '0.7', 
      '--node_dis', '1', '--split_label', '1']
    )


# mkdir to save the datalists
if not os.path.exists(root_lmdb):
  os.mkdir(root_lmdb)
path_datalist = os.path.join(root_lmdb, 'datalist')
if not os.path.exists(path_datalist):
  os.mkdir(path_datalist)

# generate datalist for octree
filename_oct_train = os.path.join(path_datalist, 'oct_train.txt')
file_oct_train = open(filename_oct_train, 'w')
filename_oct_train_aug = os.path.join(path_datalist, 'oct_train_aug.txt')
file_oct_train_aug = open(filename_oct_train_aug, 'w')
filename_oct_test = os.path.join(path_datalist, 'oct_test.txt')
file_oct_test = open(filename_oct_test, 'w')

for i in range(0, len(category)):
  path_img  = os.path.join(root_img, category[i])

  filename_img = []
  if process_img:
    filename_img = sorted(os.listdir(path_img))

  path_point = os.path.join(root_pc, category[i], 'points')
  filename_pc = sorted(os.listdir(path_point))

  path_octree = os.path.join(root_pc, category[i], 'octree')
  filename_octree = sorted(os.listdir(path_octree))

  # join filenames where render view and octree present
  filename = []
  if process_img:
    filename = [val for val in filename_img if val + '.points.ply' in filename_pc]
  else:
    filename = [val for val in filename_pc if '%s_7_2_000.octree' % val[:-7] in filename_octree]

  for item in filename[:int(len(filename) * 0.8)]:
    pass
    file_oct_train.write('%s/octree/%s_7_2_000.octree %d\n' % (category[i], item[:-7], i))

  for item in filename[int(len(filename) * 0.8):]:
    file_oct_test.write('%s/octree/%s_7_2_000.octree %d\n' % (category[i], item[:-7], i))

  for item in filename[:int(len(filename) * 0.8)]:
    file_oct_train_aug.write('%s/octree/%s_7_2_000.octree %d\n' % (category[i], item[:-7], i))
    file_oct_train_aug.write('%s/octree/%s_7_2_001.octree %d\n' % (category[i], item[:-7], i))
    file_oct_train_aug.write('%s/octree/%s_7_2_011.octree %d\n' % (category[i], item[:-7], i))

file_oct_train.close()
file_oct_test.close()
file_oct_train_aug.close()

print('Generate octree lmdb ...')
# generate lmdb for octree
# print(
subprocess.check_call(
  [convert_octree, root_pc+'/', filename_oct_train, root_lmdb+'/oct_train_lmdb']
)
# print(
subprocess.check_call(
  [convert_octree, root_pc+'/', filename_oct_train_aug, root_lmdb+'/oct_train_aug_lmdb']
)
# print(
subprocess.check_call(
  [convert_octree, root_pc+'/', filename_oct_test, root_lmdb+'/oct_test_lmdb']
)

# generate datalist for image
if process_img:
  filename_oct_train = os.path.join(path_datalist, 'oct_train_shuffle.txt')
  filename_img_train = os.path.join(path_datalist, 'img_train_shuffle.txt')
  file_img_train = open(filename_img_train, 'w')
  file_oct_train = open(filename_oct_train, 'r')
  lines = file_oct_train.readlines()
  rand_perm = []
  view_num = 24
  for i in range(0, len(lines)):
    rand_perm.append(numpy.random.permutation(view_num))
  for v in range(0, view_num):
    for i in range(0, len(lines)):
      line_img = lines[i].replace('/octree/', '/') \
                        .replace('.points_7_2_000.octree', '/rendering/%02d.png' % rand_perm[i][v])
      file_img_train.write(line_img)
  file_oct_train.close()
  file_img_train.close()

  filename_oct_test = os.path.join(path_datalist, 'oct_test_shuffle.txt')
  filename_img_test = os.path.join(path_datalist, 'img_test_shuffle.txt')
  file_img_test = open(filename_img_test, 'w')
  file_oct_test = open(filename_oct_test, 'r')
  lines = file_oct_test.readlines()
  for line in lines:
    line_img = line.replace('/octree/', '/') \
                  .replace('.points_7_2_000.octree', '/rendering/00.png')
    file_img_test.write(line_img)
  file_img_test.close()
  file_oct_test.close()


  # generate lmdb for image2shape
  print('Generate image lmdb ...')
  # print(
  subprocess.check_call(
    [convert_image, root_img+'/', filename_img_train, root_lmdb+'/img_train_lmdb', 
    '--noshuffle', '--check_size']
  )
  # print(
  subprocess.check_call(
    [convert_image, root_img+'/', filename_img_test, root_lmdb+'/img_test_lmdb', 
    '--noshuffle', '--check_size']
  )
