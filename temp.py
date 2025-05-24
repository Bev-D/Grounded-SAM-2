def process_video_frames(args_dict, step=20, text="car. road. Vegetation.", PROMPT_TYPE_FOR_VIDEO="mask"):
    video_predictor = args_dict['video_predictor']
    image_predictor = args_dict['image_predictor']
    processor = args_dict['processor']
    grounding_model = args_dict['grounding_model']
    device = args_dict['device']
    video_dir = args_dict['video_dir']
    frame_names = args_dict['frame_names']
    mask_data_dir = args_dict['mask_data_dir']
    json_data_dir = args_dict['json_data_dir']

    inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
    sam2_masks = MaskDictionaryModel()
    objects_count = 0

    video_segments_all = {}

    for start_frame_idx in range(0, len(frame_names), step):
        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

        image = Image.open(img_path)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        image_predictor.set_image(np.array(image.convert("RGB")))
        input_boxes = results[0]["boxes"]
        OBJECTS = results[0]["labels"]

        if input_boxes.shape[0] != 0:
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 2:
                masks = masks[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(device),
                box_list=torch.tensor(input_boxes),
                label_list=OBJECTS
            )

            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
        else:
            mask_dict = sam2_masks
            mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=frame_names[start_frame_idx:start_frame_idx+step])
            continue

        if len(mask_dict.labels) == 0:
            mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=frame_names[start_frame_idx:start_frame_idx+step])
            continue

        video_predictor.reset_state(inference_state)
        for object_id, object_info in mask_dict.labels.items():
            video_predictor.add_new_mask(inference_state, start_frame_idx, object_id, object_info.mask)

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionaryModel()
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)
                object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id))
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                frame_masks.mask_name = f"mask_{frame_names[out_frame_idx].split('.')[0]}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        # Save results
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id
            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            json_data = frame_masks_info.to_dict()
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)

        video_segments_all.update(video_segments)

    return video_segments_all
def setup_directories_and_frames(video_dir, output_dir):
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return {
        'frame_names': frame_names,
        'mask_data_dir': mask_data_dir,
        'json_data_dir': json_data_dir,
        'result_dir': result_dir
    }
def initialize_environment():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return {
        'video_predictor': video_predictor,
        'image_predictor': image_predictor,
        'processor': processor,
        'grounding_model': grounding_model,
        'device': device
    }
