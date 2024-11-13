import sentry_sdk
import boto3
import os
import sys
from argparse import Namespace
import time
import json
import base64
import shutil


from kohya_ss.kohya_gui import dreambooth_folder_creation_gui
from helper import getSizeInBytes, deleteOldFiles, validateTrainingMessage

from constants import MODEL_VERSION_DETAILS, EVENT_TRAIN, EVENT_DELETE_LORAS


def listFilesRecursively(directory):
    filesToReturn = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filesToReturn.append(os.path.join(root, file))
    return filesToReturn


sentry_sdk.init(
    environment=os.environ.get("ENVIRONMENT", "development"),
    dsn="https://a752d553776546f480c2af87d54358e1@o992195.ingest.sentry.io/4505565781688320",
    traces_sample_rate=0.2,
)

project_id = "relove-cloud"

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
aws_region = os.environ["AWS_REGION"]
sqs_train_queue_url = os.environ["SQS_TRAIN_QUEUE_URL"]
sqs_status_queue_url = os.environ["SQS_STATUS_QUEUE_URL"]

s3_bucket_images = os.environ["S3_BUCKET_IMAGES"]
s3_bucket_models = os.environ["S3_BUCKET_MODELS"]

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

sqs = session.client('sqs')
s3 = session.client('s3')

currentDir = os.path.dirname(os.path.abspath(__file__))


class CustomError(Exception):
    pass


if (not os.path.exists(os.path.join(currentDir, 'dataset'))):
    raise CustomError("DATASET_DIRECTORY_DOES_NOT_EXIST")

trainDataDir = os.path.join(currentDir, 'dataset', 'training')
os.makedirs(trainDataDir, exist_ok=True)

logDataDir = os.path.join(currentDir, 'dataset', 'logs')
os.makedirs(logDataDir, exist_ok=True)

# path of the NFS mounted volume - /mnt/shared_nfs   (only for GKE)
sharedLoraDir = os.path.join(os.path.abspath(os.sep), "mnt", "shared_nfs")
sharedLoraDirMaxSize = 2000 * 1000 * 1000 * 1000  # approx 2 TB
LORA_DELETION_TIMEOUT = 5 * 60  # 5mins - used for message visibility timeout



messages = [{'MessageId': '37a5f441-184d-4a79-983f-d493501d76ad', 'ReceiptHandle': 'AQEBNiU2XyMMLBybkf3chQLCdhHfL7gVYC5gPx5zEMWiOlbGSAU2/QJqdOrMTUBOAKD5/2f2+Gxd2ldqEGYjcTCU84YwSrozhe79iMjXCM3bUjsmT1C+Ttcd67jU1H/BGiyf/v8s4dKLMIReo4NaBT54V3KsReeYIL6gVqBsN08AOj0JkHsPOd/64EQ7J0Zci+LAXvXs37SgD+x05qG551SEqcy9VaaydS+dIKu22d75RAldM7nM5alEMsVFeygoSEPEIgMs4MPztFEoaWTyhTIXh0GHVMecLQxSWeF1LBLxh3YCJ3dxKvbiIPPogAI29qoNr0p2qMVtijSxDl4sPvM+D58E0pou/G0pJSx50HKjw4msm3MOg5PCtQNgaR7HxDy1U/BGhVwr6uXxa4WFqSRoJQ==', 'MD5OfBody': '2c3d93a2f1ab480ccb34436d9c39be4f', 'Body': '{"modelVersion":"flux-kohya","trainingJobId":"123","inputVersion":"t4","instancePrompt":"esk","classPrompt":"dress","trainingImagesRepeatInput":40,"productId":38,"productName":"ecru","keyword":"esk","data":[{"caption":"esk dress, white background","objectKey":"38/segmented_bulk_low_res_clypptfrm00052v6xvmcofzmj_RAMvP7wKpb0_c12880.png","image":""},{"caption":"esk dress, white background","objectKey":"38/segmented_bulk_low_res_clyppvcb100072v6xlyaa4xov_yAIqWsTduFU_eed2a3.png","image":""}],"parameters":{"console_log_level":null,"console_log_file":null,"console_log_simple":false,"v2":false,"v_parameterization":false,"pretrained_model_name_or_path":"/mnt/shared_storage/models/unet/flux1-dev.safetensors","tokenizer_cache_dir":null,"train_data_dir":null,"dataset_config": null, "cache_info":false,"shuffle_caption":false,"caption_separator":",","caption_extension":".txt","caption_extention":null,"keep_tokens":0,"keep_tokens_separator":"","secondary_separator":null,"enable_wildcard":false,"caption_prefix":null,"caption_suffix":null,"color_aug":false,"flip_aug":false,"face_crop_aug_range":null,"random_crop":false,"debug_dataset":false,"resolution":"1024,1024","cache_latents":false,"vae_batch_size":1,"cache_latents_to_disk":true,"enable_bucket":false,"min_bucket_reso":128,"max_bucket_reso":2048,"bucket_reso_steps":64,"bucket_no_upscale":false,"token_warmup_min":1,"token_warmup_step":0,"alpha_mask":false,"dataset_class":null,"caption_dropout_rate":0,"caption_dropout_every_n_epochs":0,"caption_tag_dropout_rate":0,"reg_data_dir":null,"in_json":null,"dataset_repeats":1,"output_dir":"/app/flux_new_lora","output_name":"4188_3200_steps_1e4_kohya","huggingface_repo_id":null,"huggingface_repo_type":null,"huggingface_path_in_repo":null,"huggingface_token":null,"huggingface_repo_visibility":null,"save_state_to_huggingface":false,"resume_from_huggingface":false,"async_upload":false,"save_precision":"bf16","save_every_n_epochs":1,"save_every_n_steps":null,"save_n_epoch_ratio":null,"save_last_n_epochs":null,"save_last_n_epochs_state":null,"save_last_n_steps":null,"save_last_n_steps_state":null,"save_state":false,"save_state_on_train_end":false,"resume":null,"train_batch_size":1,"max_token_length":null,"mem_eff_attn":false,"torch_compile":false,"dynamo_backend":"inductor","xformers":false,"sdpa":true,"vae":null,"max_train_steps":1600,"max_train_epochs":20,"max_data_loader_n_workers":2,"persistent_data_loader_workers":true,"seed":42,"gradient_checkpointing":true,"gradient_accumulation_steps":1,"mixed_precision":"bf16","full_fp16":false,"full_bf16":false,"fp8_base":true,"ddp_timeout":null,"ddp_gradient_as_bucket_view":false,"ddp_static_graph":false,"clip_skip":null,"logging_dir":null,"log_with":null,"log_prefix":null,"log_tracker_name":null,"wandb_run_name":null,"log_tracker_config":null,"wandb_api_key":null,"log_config":false,"noise_offset":null,"noise_offset_random_strength":false,"multires_noise_iterations":null,"ip_noise_gamma":null,"ip_noise_gamma_random_strength":false,"multires_noise_discount":0.3,"adaptive_noise_scale":null,"zero_terminal_snr":false,"min_timestep":null,"max_timestep":null,"loss_type":"l2","huber_schedule":"snr","huber_c":0.1,"lowram":false,"highvram":true,"sample_every_n_steps":null,"sample_at_first":false,"sample_every_n_epochs":null,"sample_prompts":null,"sample_sampler":"ddim","config_file":null,"output_config":false,"metadata_title":null,"metadata_author":null,"metadata_description":null,"metadata_license":null,"metadata_tags":null,"prior_loss_weight":1,"conditioning_data_dir":null,"masked_loss":false,"deepspeed":false,"zero_stage":2,"offload_optimizer_device":null,"offload_optimizer_nvme_path":null,"offload_param_device":null,"offload_param_nvme_path":null,"zero3_init_flag":false,"zero3_save_16bit_model":false,"fp16_master_weights_and_gradients":false,"optimizer_type":"adamw","use_8bit_adam":false,"use_lion_optimizer":false,"learning_rate":0.0001,"max_grad_norm":1,"optimizer_args":null,"lr_scheduler_type":"","lr_scheduler_args":null,"lr_scheduler":"constant","lr_warmup_steps":0,"lr_scheduler_num_cycles":1,"lr_scheduler_power":1,"fused_backward_pass":false,"min_snr_gamma":null,"scale_v_pred_loss_like_noise_pred":false,"v_pred_like_loss":null,"debiased_estimation_loss":false,"weighted_captions":false,"cpu_offload_checkpointing":false,"no_metadata":false,"save_model_as":"safetensors","unet_lr":null,"text_encoder_lr":null,"fp8_base_unet":false,"sdxl":false,"network_weights":null,"network_module":"networks.lora_flux","network_dim":32,"network_alpha":1,"network_dropout":null,"network_args":null,"network_train_unet_only":false,"network_train_text_encoder_only":false,"training_comment":null,"dim_from_weights":false,"scale_weight_norms":null,"base_weights":null,"base_weights_multiplier":null,"no_half_vae":false,"skip_until_initial_step":false,"initial_epoch":null,"initial_step":null,"clip_l":"/mnt/shared_storage/models/clip/clip_l.safetensors","t5xxl":"/mnt/shared_storage/models/clip/t5xxl_fp16.safetensors","ae":"/mnt/shared_storage/models/vae/ae.safetensors","t5xxl_max_token_length":null,"apply_t5_attn_mask":false,"cache_text_encoder_outputs":true,"cache_text_encoder_outputs_to_disk":true,"text_encoder_batch_size":null,"disable_mmap_load_safetensors":false,"weighting_scheme":"null","logit_mean":0,"logit_std":1,"mode_scale":1.29,"guidance_scale":1,"timestep_sampling":"shift","sigmoid_scale":1,"model_prediction_type":"raw","discrete_flow_shift":3.1582,"split_mode":false}}', 'MD5OfMessageAttributes': 'ac7985e6b1732e3f52d10247b82cb7d7', 'MessageAttributes': {'EventName': {'StringValue': 'train', 'DataType': 'String'}}}]

# messages = str(messages_n)
if not messages or len(messages) == 0:
    print("no messages found")
    exit()

try:
    message = messages[0]
    trainingJobId = ""
    messageReceiptHandle = ""
    messageBody = ""
    deletedLoras = []
    timestamp = ""
    messageEvent = ""

    messageBody = message['Body']
    messageAttributes = message['MessageAttributes']
    messageReceiptHandle = message['ReceiptHandle']
    messageEvent = messageAttributes['EventName']['StringValue']

    print('Received Message:', messageBody, messageAttributes)

    if (messageEvent == EVENT_TRAIN):
        print("Train request received")

        messageBodyObj = json.loads(messageBody)  # convert json string to dictionary

        timestamp = str(int(time.time()))
        print("timestamp", timestamp)

        # raises exception if the message body is not valid
        validateTrainingMessage(messageBodyObj)

        trainingJobId = messageBodyObj['trainingJobId']
        inputVersion = messageBodyObj['inputVersion']

        productId = messageBodyObj['productId']
        productName = messageBodyObj['productName']
        instancePrompt = messageBodyObj['instancePrompt']
        classPrompt = messageBodyObj['classPrompt']
        trainingImagesRepeatInput = messageBodyObj['trainingImagesRepeatInput']

        modelVersion = messageBodyObj['modelVersion']

        inputData = messageBodyObj['data']
        _parameters = messageBodyObj['parameters']

        enable_gradient_checkpointing = os.environ.get("ENABLE_GRADIENT_CHECKPOINTING",
                                                       _parameters.get("gradient_checkpointing", False))

        parameters = {  # default parameters
            "caption_prefix": None,
            "caption_suffix": None,
            "ip_noise_gamma": None,
            "metadata_title": None,
            "metadata_author": None,
            "metadata_description": None,
            "metadata_license": None,
            "metadata_tags": None,
            "v_pred_like_loss": None,
            "gradient_checkpointing": enable_gradient_checkpointing in [True, "True"]
        }

        parameters.update(_parameters)

        selectedModel = MODEL_VERSION_DETAILS[modelVersion]
        print("\n\n***********************selected model***************************\n\n")
        print(selectedModel)
        print("\n\n***********************selected model***************************\n\n")


        productId = str(productId)

        # create a dir to save images and captions
        inputDataDir = os.path.join(trainDataDir, timestamp, "input")
        print(inputDataDir)
        os.makedirs(inputDataDir, exist_ok=True)

        # create a dir to save prepared training data
        organizedDataDir = os.path.join(trainDataDir, timestamp, "data")
        print(organizedDataDir)
        os.makedirs(organizedDataDir, exist_ok=True)

        # create a dir to save the model
        modelDir = os.path.join(trainDataDir, timestamp, "model")
        print(modelDir)
        os.makedirs(modelDir, exist_ok=True)

        # create a dir for logs
        currLogDir = os.path.join(logDataDir, productName + "-" + inputVersion)
        print(currLogDir)
        os.makedirs(currLogDir, exist_ok=True)

        if os.path.exists(sharedLoraDir) and os.path.isdir(sharedLoraDir):
            # check shared-lora-disk size
            sharedLoraDirCurrSize = getSizeInBytes(sharedLoraDir)
            print("current lora disk size", sharedLoraDirCurrSize)

            # necessaryFiles = ["index.html", "lost+found"]   #important files and directories in the disk

            # if sharedLoraDirCurrSize > sharedLoraDirMaxSize:
            #     deletedLoras = deleteOldFiles(sharedLoraDir, sharedLoraDirCurrSize - sharedLoraDirMaxSize, necessaryFiles)
            #     print("deleted Loras", deletedLoras)

            #     if len(deletedLoras) > 0:
            #         messageSentResponse = sqs.send_message(
            #             QueueUrl=sqs_status_queue_url,
            #             MessageBody=str({
            #                 "deletedLoras": deletedLoras
            #             }),
            #             MessageAttributes={
            #                 'EventName': {
            #                     'DataType': 'String',
            #                     'StringValue': 'loras-deleted'
            #                 }
            #             }
            #         )
            if sharedLoraDirCurrSize > sharedLoraDirMaxSize:
                sentry_sdk.capture_message("No space left on shared LoRA disk", "Warning")
            elif sharedLoraDirCurrSize > 0.9 * sharedLoraDirMaxSize:
                sentry_sdk.capture_message("10% space left on shared LoRA disk", "Warning")
            elif sharedLoraDirCurrSize > 0.8 * sharedLoraDirMaxSize:
                sentry_sdk.capture_message("20% space left on shared LoRA disk", "Warning")

        for i in range(len(inputData)):
            current = inputData[i]
            imageUrl = current['image']
            fileName = 'file-' + str(i)

            objectKey = current['objectKey']
            print(f"objectjey :{objectKey}")
            print(type(objectKey))
            extension = objectKey.split(".")[-1]

            print(imageUrl, fileName, extension, objectKey)

            # create caption file
            f = open(os.path.join(inputDataDir, fileName + ".txt"), "w")
            f.write(current['caption'])
            f.close()

            # download images from s3
            s3.download_file(
                s3_bucket_images,
                objectKey,
                os.path.join(inputDataDir, fileName + "." + extension)
            )

        dreambooth_folder_creation_gui.dreambooth_folder_preparation(
            inputDataDir,
            trainingImagesRepeatInput,
            instancePrompt,
            "",
            0,
            classPrompt,
            organizedDataDir
        )

        print("training data created")

        if not os.path.exists(os.path.join(organizedDataDir, "img")):
            raise CustomError("IMAGE_DIRECTORY_DOES_NOT_EXISTS")

        parameters['train_data_dir'] = os.path.join(organizedDataDir, "img")
        parameters['output_dir'] = modelDir
        parameters['output_name'] = productName + "-" + inputVersion
        parameters['logging_dir'] = currLogDir

        parameters['pretrained_model_name_or_path'] = selectedModel["name"]
        parameters['wandb_run_name'] = f"{productId}-{inputVersion}"

        args = Namespace(**parameters)
        print(args)

        trainer = selectedModel["trainer"]
        trainer.train(args)

        # check if the model directory is empty
        file_list = os.listdir(modelDir)
        if len(file_list) == 0:
            raise CustomError("SAFETENSORS_FILE_NOT_CREATED")

        # upload the safetensor files to s3

        modelObjectKeys = []

        for sourceFilePath in listFilesRecursively(modelDir):
            # Full path of the file on the local machine
            print("source file path for file in output dir - ", sourceFilePath)
            print(f"source filepath:{type(sourceFilePath)}")
            extension = sourceFilePath.split(".")[-1]
            if (extension.strip() != "safetensors"):
                continue

            fileName = os.path.basename(sourceFilePath)
            print("file name of safetensors file - ", fileName)

            modelObjectKeys.append(fileName)

            # Upload the file to S3
            print("Uploading file to s3")
            s3.upload_file(sourceFilePath, s3_bucket_models, fileName)

            if os.path.exists(sharedLoraDir) and os.path.isdir(sharedLoraDir):
                print("copying file to shared LoRA directory")
                shutil.copy(sourceFilePath, sharedLoraDir)

        # Send the message to the queue with the specified attributes
        

    elif (messageEvent == EVENT_DELETE_LORAS):
        print("Loras deletion request received")

        messageBodyObj = json.loads(messageBody)  # convert json string to dictionary



        lorasToBeDeleted = messageBodyObj['lorasToBeDeleted']

        if not isinstance(lorasToBeDeleted, list) or len(lorasToBeDeleted) == 0:
            raise ValueError("LORAS_TO_BE_DELETED_NOT_FOUND")

        failedToDelete = []
        lorasDeleted = []

        files = os.listdir(sharedLoraDir)

        print(lorasToBeDeleted)

        for lora in lorasToBeDeleted:
            if lora not in files:
                failedToDelete.append(lora)
            filePath = os.path.join(sharedLoraDir, lora)
            os.remove(filePath)
            lorasDeleted.append(lora)

        print(lorasDeleted)
        print(failedToDelete)

        if len(lorasDeleted) > 0:
            print("lora")
        if len(failedToDelete) > 0:
            with sentry_sdk.push_scope() as scope:
                scope.set_extra("Failed to delete loras (based on event)", str(dict({
                    "ReceiptHandle": messageReceiptHandle,
                    "failedToDelete": failedToDelete,
                    "message": messageBody if isinstance(messageBody, str) else ""
                })))

                sentry_sdk.capture_message("Failed to delete loras", "Error")


except Exception as e:
    print("****Exception****", e)
    print(messageBody)
    with sentry_sdk.push_scope() as scope:
        scope.set_extra("message", str(dict({
            "failedEvent": messageEvent,
            "trainingJobId": trainingJobId,
            "ReceiptHandle": messageReceiptHandle,
            "message": messageBody if isinstance(messageBody, str) else ""
        })))

        sentry_sdk.capture_message(e, "Error")

    if messageEvent == EVENT_TRAIN:
        # send failed reason as a message
        print("failed")
finally:
    # Delete the message from the queue
    
    deleteDirectory = os.path.join(trainDataDir, timestamp)
    print(deleteDirectory)
    if timestamp != "" and os.path.isdir(deleteDirectory):
        shutil.rmtree(deleteDirectory)
