if [ ! -d "coco_dataset" ]
then
  mkdir coco_dataset
  cd coco_dataset
  wget -O- http://images.cocodataset.org/zips/train2017.zip | bsdtar -xf-
  wget -O- http://images.cocodataset.org/zips/test2017.zip | bsdtar -xf-
  wget -O- http://images.cocodataset.org/zips/val2017.zip | bsdtar -xf-
  wget -O- http://images.cocodataset.org/annotations/annotations_trainval2017.zip | bsdtar -xf-
  wget -O- http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip | bsdtar -xf-
  wget -O- http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip | bsdtar -xf-
else
  echo "Dataset folder exists."
fi