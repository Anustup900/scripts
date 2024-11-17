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

response = sqs.receive_message(
    QueueUrl=sqs_train_queue_url,
    MaxNumberOfMessages=1,
    MessageAttributeNames=['All'],
    WaitTimeSeconds=20  # Long polling to reduce API calls and improve responsiveness
)

messages = response.get('Messages', [])

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

        # change message visibility
        messageVisibilityResponse = sqs.change_message_visibility(
            QueueUrl=sqs_train_queue_url,
            ReceiptHandle=messageReceiptHandle,
            VisibilityTimeout=selectedModel["messageTimeout"]
        )

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
        if modelVersion == "flux-kohya":
            modelDir = "/app/flux_new_lora"
        else:
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
        if modelVersion == "flux-kohya":
            additional_json = {
                "output_name": productName + "-" + inputVersion
            }
            parameters.update(additional_json)
        else:
            parameters['output_name'] = productName + "-" + inputVersion
        parameters['logging_dir'] = currLogDir

        parameters['pretrained_model_name_or_path'] = selectedModel["name"]
        parameters['wandb_run_name'] = f"{productId}-{inputVersion}"
        print(parameters)
        # if selectedModel["model_name"] == "flux":
        parameters.update(selectedModel["training_args"])
        print(parameters)
        # else:
        #     parameters.update()
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
        messageSentResponse = sqs.send_message(
            QueueUrl=sqs_status_queue_url,
            MessageBody=str({
                "trainingJobId": trainingJobId,
                "status": "complete",
                "modelObjectKeys": modelObjectKeys
            }),
            MessageAttributes={
                'EventName': {
                    'DataType': 'String',
                    'StringValue': 'update-status'
                }
            }
        )

        print(f"Message sent. Message ID: {messageSentResponse['MessageId']}")

    elif (messageEvent == EVENT_DELETE_LORAS):
        print("Loras deletion request received")

        messageBodyObj = json.loads(messageBody)  # convert json string to dictionary

        # change message visibility
        messageVisibilityResponse = sqs.change_message_visibility(
            QueueUrl=sqs_train_queue_url,
            ReceiptHandle=messageReceiptHandle,
            VisibilityTimeout=LORA_DELETION_TIMEOUT
        )

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
            messageSentResponse = sqs.send_message(
                QueueUrl=sqs_status_queue_url,
                MessageBody=str({
                    "deletedLoras": lorasDeleted
                }),
                MessageAttributes={
                    'EventName': {
                        'DataType': 'String',
                        'StringValue': 'loras-deleted'
                    }
                }
            )
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
        messageSentResponse = sqs.send_message(
            QueueUrl=sqs_status_queue_url,
            MessageBody=str({
                "trainingJobId": trainingJobId,
                "status": "failed",
                "reason": base64.b64encode(str(e).encode('utf-8')).decode('utf-8')  # catch again
            }),
            MessageAttributes={
                'EventName': {
                    'DataType': 'String',
                    'StringValue': 'update-status'
                }
            }
        )

        print(f"Failed Message sent. Message ID: {messageSentResponse['MessageId']}")
finally:
    # Delete the message from the queue
    sqs.delete_message(
        QueueUrl=sqs_train_queue_url,
        ReceiptHandle=messageReceiptHandle
    )

    deleteDirectory = os.path.join(trainDataDir, timestamp)
    print(deleteDirectory)
    if timestamp != "" and os.path.isdir(deleteDirectory):
        shutil.rmtree(deleteDirectory)
