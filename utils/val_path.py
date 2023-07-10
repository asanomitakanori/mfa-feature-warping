from glob import glob
from os.path import join
from hydra.utils import to_absolute_path as abs_path


# あとでlistを保存して呼び込む形に変更
def val_path(eval_imgs_path):
    valdata_list = []
    folder = sorted(glob(join(abs_path(eval_imgs_path) + "/*")))
    for i in folder:
        valdata_list.append(glob(i + "/*"))
    valdata_list = [x for row in valdata_list for x in row]
    valdata_list.sort()
    val_clip = []
    test_step = 0
    for idx in range(len(valdata_list)):
        if (test_step) % 1 == 0:
            clip, imgs, seqids = [], [], []
            # for i in range(2):
            cur_img = valdata_list[max(0, idx)]
            seqid = cur_img[-10:-7]
            imgs.append(cur_img)
            seqids.append(seqid)

            for i in range(2 - 1):
                cur_img = imgs[i]
                seq_id1 = seqids[i]
                clip.append(cur_img)

            val_clip.append(clip[::-1])
    return val_clip
