def visualize(f, axes_images, axes_masks, axes_pred, predicted, inputs):
  # class_predictions = predicted['class_predictions']
  # bbox_predictions = predicted['bbox_predictions'] 
  segmentation_masks = predicted['segmentation_masks'].cpu()

  batch_tensor_bhwc = inputs.rearrange("bs c h w -> bs h w c")

  for im, ax in zip(batch_tensor_bhwc.data, axes_images):
    ax.imshow(im)

  for mask, ax in zip(batch_tensor_bhwc.mask, axes_masks):
    ax.imshow(mask.float().squeeze(-1))

  for mask, ax in zip(segmentation_masks.detach(), axes_pred):
    ax.imshow(mask[0:3].permute(1,2,0))

  f.suptitle("bs: {}, height: {}, width: {}, channels: {}".format(*batch_tensor_bhwc.shape));

  return f


def draw_box(ax, box, linewidth=1, edgecolor='b', facecolor='none'):
  cx,cy,w,h = box_xyxy_to_cxcywh(box).unbind(-1)
  anchor_point = (cx-w/2, cy-h/2)
  rect = patches.Rectangle(anchor_point, w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
  ax.add_patch(rect)