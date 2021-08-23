from torch.utils.tensorboard import SummaryWriter
import torch
tb = SummaryWriter()
epoch = 0
idx = 0
class_labels_to_ids = {
   1:0,    2:1,    3:2,    4:3,    5:4,    6:5,    7:6,    8:7,    9:8,   10:9, 
  11:10,  13:11,  14:12,  15:13,  16:14,  17:15,  18:16,  19:17,  20:18,  21:19, 
  22:20,  23:21,  24:22,  25:23,  27:24,  28:25,  31:26,  32:27,  33:28,  34:29,
  35:30,  36:31,  37:32,  38:33,  39:34,  40:35,  41:36,  42:37,  43:38,  44:39,
  46:40,  47:41,  48:42,  49:43,  50:44,  51:45,  52:46,  53:47,  54:48,  55:49,
  56:50,  57:51,  58:52,  59:53,  60:54,  61:55,  62:56,  63:57,  64:58,  65:59,
  67:60,  70:61,  72:62,  73:63,  74:64,  75:65,  76:66,  77:67,  78:68,  79:69,
  80:70,  81:71,  82:72,  84:73,  85:74,  86:75,  87:76,  88:77,  89:78,  90:79,
}
class_names = {
   0:'person',        1:'bicycle',    2:'car',            3:'motorcycle',  4:'airplane',      5:'bus',             6:'train',        7:'truck',        8:'boat',            9:'traffic light',
  10:'fire hydrant', 11:'stop sign', 12:'parking meter', 13:'bench',      14:'bird',         15:'cat',            16:'dog',         17:'horse',       18:'sheep',          19:'cow',
  20:'elephant',     21:'bear',      22:'zebra',         23:'giraffe',    24:'backpack',     25:'umbrella',       26:'handbag',     27:'tie',         28:'suitcase',       29:'frisbee',
  30:'skis',         31:'snowboard', 32:'sports ball',   33:'kite',       34:'baseball bat', 35:'baseball glove', 36:'skateboard',  37:'surfboard',   38:'tennis racket',  39:'bottle',
  40:'wine glass',   41:'cup',       42:'fork',          43:'knife',      44:'spoon',        45:'bowl',           46:'banana',      47:'apple',       48:'sandwich',       49:'orange',
  50:'broccoli',     51:'carrot',    52:'hot dog',       53:'pizza',      54:'donut',        55:'cake',           56:'chair',       57:'couch',       58:'potted plant',   59:'bed',
  60:'dining table', 61:'toilet',    62:'tv',            63:'laptop',     64:'mouse',        65:'remote',         66:'keyboard',    67:'cell phone',  68:'microwave',      69:'oven',
  70:'toaster',      71:'sink',      72:'refrigerator',  73:'book',       74:'clock',        75:'vase',           76:'scissors',    77:'teddy bear',  78:'hair drier',     79:'toothbrush'
}
NIL_CLASS_ID = 80
num_classes = len(class_labels_to_ids) + 1
label2tensor = torch.eye(num_classes)