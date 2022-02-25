anno_dir="/home/myjian/Workespace/PythonProject/Dataset/VOCdevkit/VOC2007/Annotations"
dataset_type="voc"
# anno_list="/home/myjian/Workespace/PythonProject/Dataset/VOCdevkit/VOC2007/ImageSets/Main/person_train.txt"
anno_list="/home/myjian/Workespace/PythonProject/Dataset/VOCdevkit/VOC2007/ImageSets/Layout/trainval.txt"
out_name="voc2coco.json"
python tools/trans2coco.py --dataset_type $dataset_type \
               --voc_anno_dir $anno_dir \
               --voc_anno_list $anno_list \
               --voc_out_name $out_name
