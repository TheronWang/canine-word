import os
import json
import random
import cv2
import torch
from transformers import AutoModelForCausalLM
# from janus.models import MultiModalityCausalLM, VLChatProcessor
# from janus.utils.io import load_pil_images
# from infer_base import inference_janus

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


def extract_video_clip(
    video_path: str,
    start_ts: float,
    end_ts: float,
    output_path: str,
    sentence_id: str,
) -> str:
    """
    Extract a video clip between start_ts and end_ts from the video file,
    and save it in the specified output folder. This function preserves
    both video and audio tracks using FFmpeg.

    Returns the path to the saved video clip.

    The output file name will be of the format:
    {sentence_id}_clip.mp4
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # First, get video duration using FFmpeg to validate timestamps
    import subprocess
    import json
    
    # Get video duration using FFprobe
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_format', video_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(probe_result.stdout)
        video_duration = float(video_info['format']['duration'])
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Could not get video duration using ffprobe: {e}")
        # Fallback: use OpenCV to get duration
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()

    # extend the start ts and end ts 3 seconds
    start_ts = max(0, start_ts - 3)
    end_ts = min(video_duration, end_ts + 3)

    # Ensure the specified timestamps are within the video duration
    if start_ts < 0 or start_ts > video_duration:
        raise ValueError("start_ts must be between 0 and the video duration.")
    if end_ts < 0 or end_ts > video_duration:
        end_ts = video_duration
    if start_ts >= end_ts:
        raise ValueError("start_ts must be less than end_ts.")

    # Construct the output filename
    output_filename = f"{sentence_id}_clip.mp4"
    output_filepath = os.path.join(output_path, output_filename)

    # Use FFmpeg to extract video clip with audio
    # -ss: start time, -t: duration, -c copy: copy streams without re-encoding
    duration = end_ts - start_ts
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y: overwrite output file without asking
        '-ss', str(start_ts),  # start time
        '-i', video_path,  # input file
        '-t', str(duration),  # duration
        '-c', 'copy',  # copy streams without re-encoding (preserves quality and audio)
        '-avoid_negative_ts', 'make_zero',  # handle negative timestamps
        output_filepath
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print(f"Saved video clip with audio from {start_ts:.2f}s to {end_ts:.2f}s to {output_filepath}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Failed to extract video clip: {e}")

    return output_filepath


def extract_all_video_clips():
    """
    Extract video clips for all dog sentences following the same logic as
    infer_dog_sentence_activities but only for video extraction.
    """
    # Load dog sentence data
    dog_sentence_file = "/home/sinong/ALP/sentence_meta.json"
    with open(dog_sentence_file) as f:
        dog_sentences = json.load(f)
    print(f"Loaded {len(dog_sentences)} dog sentences")

    # Load raw video data
    raw_video_dict_path = "/home/sinong/ALP/AniLing/raw_video_path.json"
    with open(raw_video_dict_path) as f:
        raw_video_dict = json.load(f)
    print(f"Loaded {len(raw_video_dict)} video paths")

    # build sentence pool
    sentence_pool = list(dog_sentences.keys())
    random.shuffle(sentence_pool)

    # if there are already extracted clips, load the tracking file
    clips_tracking_file = "extracted_clips_tracking2.json"
    if os.path.exists(clips_tracking_file):
        with open(clips_tracking_file) as f:
            extracted_clips = json.load(f)
        print(f"Loaded {len(extracted_clips)} already extracted clips")
    else:
        extracted_clips = dict()

    # Video clips output folder (similar to frames folder)
    clips_output_path = "/data/_assets/dog_sentence_clips2"

    # Start the extraction
    processed_count = 0
    for sentence_id in sentence_pool:
        if sentence_id in extracted_clips:
            continue

        # get the meta data
        video_id = sentence_id.split("+")[1]
        try:
            video_path = raw_video_dict[video_id]
        except KeyError:
            print(f"Video {video_id} not found in the raw video data")
            continue
        
        start_time = dog_sentences[sentence_id]["start"]
        end_time = dog_sentences[sentence_id]["end"]
        sentence_audio_path = dog_sentences[sentence_id]["path"]

        # Extract video clip
        try:
            clip_path = extract_video_clip(
                video_path,
                start_time,
                end_time,
                clips_output_path,
                sentence_id,
            )
            
            extracted_clips[sentence_id] = {
                "audio_path": sentence_audio_path,
                "video_clip_path": clip_path,
                "start_time": start_time,
                "end_time": end_time,
                "video_id": video_id,
            }
            
            processed_count += 1
            print(f"Processed {processed_count} clips. Current: {sentence_id}")
            
        except Exception as e:
            print(f"Error extracting video clip for sentence {sentence_id}: {e}")
            continue

        # save the tracking file for every 100 clips
        if processed_count % 100 == 0:
            with open(clips_tracking_file, "w") as f:
                json.dump(extracted_clips, f)
            print(f"Saved tracking file with {len(extracted_clips)} clips")

    # Save the final tracking file
    with open(clips_tracking_file, "w") as f:
        json.dump(extracted_clips, f)
    
    print(f"Completed! Extracted {len(extracted_clips)} video clips to {clips_output_path}")


def test_inference_janus():
    model_path = "deepseek-ai/Janus-Pro-7B"
    question = [
        "Below are five frames from the same video clip, showing a dog in a"
        " specific scenario <image_placeholder>, <image_placeholder>,"
        " <image_placeholder>, <image_placeholder>, <image_placeholder>."
        " Consider all five images collectively as representing a single"
        " event. Select exactly ONE environment label and ONE activity label"
        " from the lists provided below that best describe this"
        " event.\nEnvironment labels: [indoor floor, sofa, bathroom, bed, near"
        " the window, near the door, on grass, on beach, near other animals,"
        " vehicle interior]\nActivity labels: [rest, running, alerting to"
        " sounds, seeking attention, playing with human, playing with toy,"
        " swimming, begging for food, displaying aggression, taking shower,"
        " exploring]\nRespond with exactly two labels:Environment:"
        " ___;Activity: ___"
    ]

    image_path = [
        "/home/sinong/ALP/AniLing/_assets/images/dog_screenshot.png",
        "/home/sinong/ALP/AniLing/_assets/images/dog_screenshot.png",
        "/home/sinong/ALP/AniLing/_assets/images/dog_screenshot.png",
        "/home/sinong/ALP/AniLing/_assets/images/dog_screenshot.png",
        "/home/sinong/ALP/AniLing/_assets/images/dog_screenshot.png",
    ]

    result = inference_janus(model_path, question, image_path)
    print(result)


def infer_dog_activities():
    # Load the word meta data
    # The words_meta dictionary structure is:
    # {
    #    "word_0": {
    #       "sentence_id_0": {
    #           "video_path": "/path/to/video",
    #           "img_sample_path": ["/path/to/img", ...],
    #           "environment": [...],
    #           "activity": [...],
    #       },
    #       "sentence_id_1": { ... },
    #    },
    #    "word_1": { ... },
    #    ...
    # }

    # get top 100 words
    word_counts_file = "/home/sinong/ALP/AniLing/word_counts.json"
    with open(word_counts_file) as f:
        word_counts = json.load(f)
    word_list = list(word_counts.keys())[:100]

    # load words_meta data
    words_meta_file = "/home/sinong/ALP/AniLing/words_meta.json"
    with open(words_meta_file) as f:
        words_meta = json.load(f)

    words_meta_path = "/home/sinong/ALP/AniLing/words_meta.json"
    with open(words_meta_path, "r") as f:
        words_meta = json.load(f)

    model_path = "deepseek-ai/Janus-Pro-7B"

    # Load the chat processor and tokenizer
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        model_path
    )
    tokenizer = vl_chat_processor.tokenizer

    # Load the multi-modality model
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Config the input

    # Prompt from the paper (Appendix B)
    question = [
        "Describe the environment and the activity of the dog in the image"
        " using a list of words, follow the format:"
        ' {"environment": [description1, description2, ...],'
        ' "activity": [description1, description2, ...]}'
    ]
    selected_word_meta = {}
    for idx, word in enumerate(word_list):
        try:
            word_meta_data = words_meta[word]
        except KeyError:
            print(f"Word {word} not found in the words_meta data")
            continue

        selected_word_meta[word] = words_meta[word].copy()
        for sentence_id, sentence_meta_data in word_meta_data.items():
            for img_path in sentence_meta_data["img_sample_path"]:
                # config the conversation
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{question}",
                        "images": [img_path],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True,
                ).to(vl_gpt.device)

                # Prepare input embeddings for the batch
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # Generate responses for all conversations at once
                outputs = vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
                answer = tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )
                try:
                    if answer.startswith("["):
                        answer = answer[2:-2]
                    answer_json = json.loads(answer)
                except json.JSONDecodeError:
                    print(f"Error parsing the answer: {answer}")
                    continue

                # Update the word meta data
                selected_word_meta[word][sentence_id]["activity"].append(
                    answer_json["activity"]
                )
                selected_word_meta[word][sentence_id]["environment"].append(
                    answer_json["environment"]
                )
        selected_word_meta[word] = word_meta_data
        print(f"Finished processing {idx} word: {word}")

    # Save the updated word meta data
    with open("selected_word_meta.json", "w") as f:
        json.dump(selected_word_meta, f)


def get_frames_from_video(
    video_path: str,
    start_ts: float,
    end_ts: float,
    num_frames: int,
    output_path: str,
    sentence_id: str,
) -> list[str]:
    """
    Extract `num_frames` randomly selected frames between start_ts
    and end_ts from the video file, and save them in the specified
    output folder.

    Returns a list of the paths to the saved frame images.

    The output file names will be of the format:
    {sentence_id}_frame_{frame_idx}.jpg
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get video properties: FPS and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # extend the start ts and end ts 3 seconds
    start_ts = max(0, start_ts - 3)
    end_ts = min(video_duration, end_ts + 3)

    # Ensure the specified timestamps are within the video duration
    if start_ts < 0 or start_ts > video_duration:
        raise ValueError("start_ts must be between 0 and the video duration.")
    if end_ts < 0 or end_ts > video_duration:
        end_ts = video_duration
    if start_ts >= end_ts:
        raise ValueError("start_ts must be less than end_ts.")

    # Randomly choose `num_frames` timestamps between start_ts and end_ts
    timestamps = [random.uniform(start_ts, end_ts) for _ in range(num_frames)]
    # Sorting timestamps can make frame seeking more efficient (optional)
    timestamps.sort()

    frames_img_paths = []

    for idx, ts in enumerate(timestamps, start=1):
        # Convert timestamp to frame index
        frame_index = int(ts * fps)
        # Set the video capture to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # Construct the output filename
            original_height, original_width = frame.shape[:2]
            # Resize the frame to 224x224
            frame = cv2.resize(frame, (224, 224))
            output_filename = f"{sentence_id}_frame_{idx}.jpg"
            output_filepath = os.path.join(output_path, output_filename)
            # Save the frame as a JPEG file
            cv2.imwrite(output_filepath, frame)
            frames_img_paths.append(output_filepath)
            print(
                f"Saved frame at {ts:.2f}s (frame {frame_index}) to"
                f" {output_filepath}"
            )
        else:
            print(
                f"Warning: Could not retrieve frame at {ts:.2f}s (frame"
                f" {frame_index})."
            )

    # Release the video capture object
    cap.release()
    return frames_img_paths


def infer_dog_sentence_activities(
    num_annotated_sentences=200000, num_frames: int = 5
):
    # Load dog sentence data
    # Randomly select 3000 sentences
    dog_sentence_file = "/home/sinong/ALP/sentence_meta.json"
    with open(dog_sentence_file) as f:
        dog_sentences = json.load(f)
    for key in dog_sentences:
        print(key, dog_sentences[key])
        break
    print(f"Loaded {len(dog_sentences)} dog sentences")

    # Load raw video data
    raw_video_dict_path = "/home/sinong/ALP/AniLing/raw_video_path.json"
    with open(raw_video_dict_path) as f:
        raw_video_dict = json.load(f)
    for key in raw_video_dict:
        print(key, raw_video_dict[key])
        break
    print(f"Loaded {len(raw_video_dict)} video paths")

    # build sentence pool and result dict
    sentence_pool = list(dog_sentences.keys())
    random.shuffle(sentence_pool)

    # if there is already annotated sentences, load them
    if os.path.exists("annotated_sentences.json"):
        with open("annotated_sentences.json") as f:
            annotated_sentences = json.load(f)
        print(f"Loaded {len(annotated_sentences)} annotated sentences")
    else:
        annotated_sentences = dict()

    # Load the chat processor and tokenizer
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        model_path
    )
    tokenizer = vl_chat_processor.tokenizer

    # Load the multi-modality model
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Config the input
    question = [
        "Below are five frames from the same video clip, showing a dog in a"
        " specific scenario <image_placeholder>, <image_placeholder>,"
        " <image_placeholder>, <image_placeholder>, <image_placeholder>."
        " Consider all five images collectively as representing a single"
        " event. Select exactly ONE environment label and ONE activity label"
        " from the lists provided below that best describe this"
        " event.\nEnvironment labels: [cage, indoor floor, bathroom, sofa&bed,"
        " on grass, on beach, outdoor ground, around the house, in a vehicle,"
        " unknown] \nActivity labels: [rest, shower, eat&drink, interact with"
        " human, interact with other animals, play toys, face out of camera,"
        " exploration, unknown]  \nRespond with exactly two labels in the"
        " <env></env> and <act></act> format:\n<env>environment label</env>"
        " <act>activity label</act>"
    ]

    # Start the inference
    while len(sentence_pool) > 0:
        # randomly select a sentence from the pool
        sentence_id = sentence_pool.pop()
        if sentence_id in annotated_sentences:
            continue

        # get the meta data
        video_id = sentence_id.split("+")[1]
        try:
            video_path = raw_video_dict[video_id]
        except KeyError:
            print(f"Video {video_id} not found in the raw video data")
            continue
        start_time = dog_sentences[sentence_id]["start"]
        end_time = dog_sentences[sentence_id]["end"]
        sentence_audio_path = dog_sentences[sentence_id]["path"]

        # Extract frames from the video
        try:
            frame_path_list = get_frames_from_video(
                video_path,
                start_time,
                end_time,
                num_frames,
                "/data/_assets/dog_sentence_frames",
                sentence_id,
            )
        except Exception as e:
            print(f"Error extracting frames for sentence {sentence_id}: {e}")
            continue
        if len(frame_path_list) != num_frames:
            print(
                f"Warning: Extracted {len(frame_path_list)} frames for"
                f" sentence {sentence_id}, expected {num_frames}"
            )
            continue
        print(
            f"Extracted {len(frame_path_list)} frames for sentence"
            f" {sentence_id}"
        )

        # config the conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{question}",
                "images": frame_path_list,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)

        # Process the batch of conversations.
        # The parameter `force_batchify=True` tells the processor to combine the inputs into a batch.
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(vl_gpt.device)

        # Prepare input embeddings for the batch
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Generate responses for all conversations at once
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        environment_description = answer.split(";")[0].split(":")[1].strip()
        activity_description = answer.split(";")[1].split(":")[1].strip()

        annotated_sentences[sentence_id] = {
            "audio_path": sentence_audio_path,
            "frames": frame_path_list,
            "environment": environment_description,
            "activity": activity_description,
            "answer": answer,
        }
        print(f"Finished processing sentence {sentence_id}")

        # save the annotated sentences for every 100 sentences
        if len(annotated_sentences) % 100 == 0:
            with open("annotated_sentences.json", "w") as f:
                json.dump(annotated_sentences, f)

    # Save the annotated sentences
    with open("annotated_sentences.json", "w") as f:
        json.dump(annotated_sentences, f)


def delete_later():
    filename = "annotated_sentences.json"
    with open(filename, "r") as f:
        annotated_sentences = json.load(f)
    print(f"Loaded {len(annotated_sentences)} annotated sentences")


def infer_dog_sentence_activities_loose(num_frames: int = 5):
    # Load dog sentence data
    # Randomly select 3000 sentences
    dog_sentence_file = "/home/sinong/ALP/sentence_meta.json"
    with open(dog_sentence_file) as f:
        dog_sentences = json.load(f)
    for key in dog_sentences:
        print(key, dog_sentences[key])
        break
    print(f"Loaded {len(dog_sentences)} dog sentences")

    # Load raw video data
    raw_video_dict_path = "/home/sinong/ALP/AniLing/raw_video_path.json"
    with open(raw_video_dict_path) as f:
        raw_video_dict = json.load(f)
    for key in raw_video_dict:
        print(key, raw_video_dict[key])
        break
    print(f"Loaded {len(raw_video_dict)} video paths")

    # build sentence pool and result dict
    sentence_pool = list(dog_sentences.keys())
    random.shuffle(sentence_pool)

    # if there is already annotated sentences, load them
    if os.path.exists("annotated_sentences_verbose.json"):
        with open("annotated_sentences_verbose.json") as f:
            annotated_sentences = json.load(f)
        print(f"Loaded {len(annotated_sentences)} annotated sentences")
    else:
        annotated_sentences = dict()

    # Load the chat processor and tokenizer
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        model_path
    )
    tokenizer = vl_chat_processor.tokenizer

    # Load the multi-modality model
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Config the input
    question = [
        "Below are five frames from a short dog bark video, depicting a dog in"
        " a specific scenario. The frames show the dog in various positions"
        " and moments of the event: <image_placeholder>, <image_placeholder>,"
        " <image_placeholder>, <image_placeholder>, <image_placeholder>."
        " Please consider these five images together as representing a single"
        " event. Based on these frames, provide a VERBOSE description with the"
        " following details:\n\n1. Environment: Describe the environment of"
        " the dog. List all the relevant environmental elements (e.g., indoor,"
        " outdoor, grassy, urban, etc.), and put them into a comma-separated"
        " list enclosed within <env></env>. Each element should be a single"
        " word or short phrase.\n\n2. Activity: Describe the activity the dog"
        " is engaged in during the event. List all relevant activities (e.g.,"
        " running, sitting, barking, etc.), and put them into a"
        " comma-separated list enclosed within <act></act>. Each element"
        " should be a single word or short phrase.\n\n3. Reason for Barking:"
        " Provide a few sentences explaining why the dog might be barking."
        " Consider the context and possible causes (e.g., excitement, warning,"
        " attention-seeking, etc.), and describe the reasoning in a clear and"
        " concise manner.\n\nOutput Format:\n- Environment: <env>word1, word2,"
        " word3, ...</env>\n- Activity: <act>action1, action2, action3,"
        ' ...</act>\n- Reason for Barking: "explanation sentence 1,'
        ' explanation sentence 2, ..."'
    ]

    # Start the inference
    while len(sentence_pool):
        # randomly select a sentence from the pool
        sentence_id = sentence_pool.pop()
        if sentence_id in annotated_sentences:
            continue

        # get the meta data
        video_id = sentence_id.split("+")[1]
        try:
            video_path = raw_video_dict[video_id]
        except KeyError:
            print(f"Video {video_id} not found in the raw video data")
            continue
        start_time = dog_sentences[sentence_id]["start"]
        end_time = dog_sentences[sentence_id]["end"]
        sentence_audio_path = dog_sentences[sentence_id]["path"]

        # Extract frames from the video
        try:
            frame_path_list = get_frames_from_video(
                video_path,
                start_time,
                end_time,
                num_frames,
                "/data/_assets/dog_sentence_frames",
                sentence_id,
            )
        except Exception as e:
            print(f"Error extracting frames for sentence {sentence_id}: {e}")
            continue
        if len(frame_path_list) != num_frames:
            print(
                f"Warning: Extracted {len(frame_path_list)} frames for"
                f" sentence {sentence_id}, expected {num_frames}"
            )
            continue
        print(
            f"Extracted {len(frame_path_list)} frames for sentence"
            f" {sentence_id}"
        )

        # config the conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{question}",
                "images": frame_path_list,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)

        # Process the batch of conversations.
        # The parameter `force_batchify=True` tells the processor to combine the inputs into a batch.
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(vl_gpt.device)

        # Prepare input embeddings for the batch
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Generate responses for all conversations at once
        # Repeat the process for 5 times
        answers = []
        for _ in range(1):
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
            answer = tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
            answers.append(answer)

        annotated_sentences[sentence_id] = {
            "audio_path": sentence_audio_path,
            "frames": frame_path_list,
            "description": answers,
        }
        # save the annotated sentences for every 100 sentences
        if len(annotated_sentences) % 100 == 0:
            with open("annotated_sentences_verbose.json", "w") as f:
                json.dump(annotated_sentences, f)

    # Save the annotated sentences
    with open("annotated_sentences_verbose.json", "w") as f:
        json.dump(annotated_sentences, f)


if __name__ == "__main__":
    # test_inference_janus()
    # infer_dog_activities()
    # infer_dog_sentence_activities()
    # infer_dog_sentence_activities_loose()
    # delete_later()
    extract_all_video_clips()
