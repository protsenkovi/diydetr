from matplotlib.pyplot import boxplot
from utils.functions import box_xyxy_to_cxcywh, uniform_shape_masks
from matplotlib import patches
from utils import config

def visualize(f, axes_images, axes_masks, axes_pred, axes_target_masks, predicted, inputs, targets):

  class_predictions = predicted['class_predictions'].cpu().detach()
  bbox_predictions = predicted['bbox_predictions'].cpu().detach()
  segmentation_masks = predicted['segmentation_masks'].cpu().detach()

  batch_tensor_bhwc = inputs.rearrange("bs c h w -> bs h w c")

  for im, ax, target in zip(batch_tensor_bhwc.data, axes_images, targets):
    im = im - im.min()
    im /= im.max()
    # im *= 255
    # print(im.min(), im.max())
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(im.shape[0], 0 )
    ax.imshow(im, origin="upper")
    ax.set_title("id: {}".format(target['image_id'].item()))

  for im, ax, class_probs, boxes, target in zip(batch_tensor_bhwc.data, axes_images, class_predictions, bbox_predictions, targets):
      cids = class_probs.argmax(-1)
      h, w = target['size']
      for cid, box in zip(cids, boxes):
        if cid != config.NIL_CLASS_ID:
          box[0::2] *= w
          box[1::2] *= h
          draw_box(ax, box, edgecolor='C{}'.format(cid), text='{}'.format(config.class_names[cid.item()]))

  if axes_masks is not None or batch_tensor_bhwc.mask is not None:
    for mask, ax in zip(batch_tensor_bhwc.mask, axes_masks):
      ax.imshow(mask.float().squeeze(-1))

  for mask, ax in zip(segmentation_masks.detach(), axes_pred):
    ax.imshow(mask[0:3].permute(1,2,0))

  target_masks = uniform_shape_masks([t['masks'] for t in targets]).float()
  target_masks_count = [t['masks'].shape[0] for t in targets]

  for meta, t_masks, t_count, ax in zip(targets, target_masks, target_masks_count, axes_target_masks):
    if t_masks.numel() != 0:
      legend_handels = []
      if t_count >= 3:
        ax.imshow(t_masks[0:3].permute(1,2,0))
        label = meta['labels'][0].item()
        cid = config.class_labels_to_ids[label]
        c_name = config.class_names[cid]
        legend_handels.append(patches.Patch(color='red', label="label: {}, id: {}, {}".format(label, cid, c_name)))

        label = meta['labels'][1].item()
        cid = config.class_labels_to_ids[label]
        c_name = config.class_names[cid]
        legend_handels.append(patches.Patch(color='green', label="label: {}, id: {}, {}".format(label, cid, c_name)))

        label = meta['labels'][2].item()
        cid = config.class_labels_to_ids[label]
        c_name = config.class_names[cid]
        legend_handels.append(patches.Patch(color='blue', label="label: {}, id: {}, {}".format(label, cid, c_name)))
      if t_count == 1:
        ax.imshow(t_masks[0:1].permute(1,2,0))

        label = meta['labels'][0].item()
        cid = config.class_labels_to_ids[label]
        c_name = config.class_names[cid]
        legend_handels.append(patches.Patch(color='red', label="label: {}, id: {}, {}".format(label, cid, c_name)))

      ax.legend(handles=legend_handels, prop={'size': 6})


  f.suptitle("epoch:{}\nbs: {}, height: {},\n width: {}, channels: {}".format(config.epoch, *batch_tensor_bhwc.shape));

  return f


def draw_box(ax, box, linewidth=1, edgecolor='b', facecolor='none', text=None, label=None, **kwargs):
  cx,cy,w,h = box.unbind(-1)
  anchor_point = (cx-w/2, cy-h/2)
  rect = patches.Rectangle(anchor_point, w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, label=label, **kwargs)
  ax.add_patch(rect)
  if text is not None:
    ax.text(cx, cy, text, color=edgecolor, bbox=dict(boxstyle="square", fc=(0., 0., 0., 0.6), ec=(1., 1., 1., 0.0)))
  ax.scatter(cx, cy, c=edgecolor, s=linewidth)