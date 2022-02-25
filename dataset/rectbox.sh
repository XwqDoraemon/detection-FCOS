input_label="/home/myjian/Workespace/PythonProject/Dataset/bdd100k/labels/det_20/det_train.json"
input_image="/home/myjian/Workespace/PythonProject/Dataset/bdd100k/images/100k/train"
output="/home/myjian/Workespace/PythonProject/Dataset/vision_bdd100k"
python dataset/rectbox.py -f $input_label -i $input_image -o $output