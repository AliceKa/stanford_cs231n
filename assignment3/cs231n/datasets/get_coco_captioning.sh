# Use curl instead of wget, it's already installed on OS X
# wget "http://cs231n.stanford.edu/coco_captioning.zip"

curl -O "http://cs231n.stanford.edu/coco_captioning.zip"
unzip coco_captioning.zip
rm coco_captioning.zip
