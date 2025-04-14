import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import os

# Download checkpoints
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts", ignore_patterns=["*virtual_tryon.pth", "*pose_transfer.pth", "stable-diffusion-xl-*"])


class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type="virtual_tryon",
        ref_acceleration=True,
        step=50,
        scale=2.5,
        seed=42,
        #vt_model_type="dress_code",
        vt_garment_type="dresses",
        vt_repaint=True,
        preprocess_garment=False
    ):
        # Open and resize the source image.
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                # preprocess_garment_image returns a 768x1024 image.
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            # Otherwise, load the reference image.
            ref_image = Image.open(ref_image_path)

        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
        mask = mask.resize((768, 1024))

        src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
        src_image_seg_array = src_image_iuv_array[:, :, 0:1]
        src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
        src_image_seg = Image.fromarray(src_image_seg_array)
        densepose = src_image_seg

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        inference = self.vt_inference_dc

        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )

        gen_image = output["generated_image"][0]

        return np.array(gen_image), np.array(mask), np.array(densepose)
    
if __name__ == "__main__":
    leffa_predictor = LeffaPredictor()
    example_dir = "./ckpts/examples"

    gen_image, mask, dense = leffa_predictor.leffa_predict(
        src_image_path="./ckpts/examples/person1/01350_00.jpg",
        ref_image_path="./ckpts/examples/garment/01449_00.jpg"
    )

    # 出力フォルダを定義
    output_dir = "/opt/artifact"
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名（例: 日時やパラメータに応じて動的にすることも可能）
    gen_path = os.path.join(output_dir, "generated.png")
    mask_path = os.path.join(output_dir, "mask.png")
    densepose_path = os.path.join(output_dir, "densepose.png")

    # NumPy配列から画像に変換して保存
    Image.fromarray(gen_image).save(gen_path)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(dense).save(densepose_path)

    print(f"Saved generated image to {gen_path}")
    print(f"Saved mask to {mask_path}")
    print(f"Saved densepose to {densepose_path}")








