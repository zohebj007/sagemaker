from sagemaker import Session
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession
import boto3
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.model import SKLearnModel

bucket='zohebmlops'
prefix = 'mlops'
input_source = sagemaker.Session().upload_data('./iris.csv', bucket=bucket, key_prefix=f'{prefix}')
train_path = f"s3://{bucket}/{prefix}/train"
test_path = f"s3://{bucket}/{prefix}/test"
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()


sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1, 
    base_job_name='mlops-sklearnprocessing'
)

# Define processing step
processing_step = ProcessingStep(
    name='PreprocessingStep',
    processor=sklearn_processor,
    code='preprocess.py',  # Path to your preprocessing script
    inputs=[
        ProcessingInput(
            source=input_source, 
            destination="/opt/ml/processing/input",
            s3_input_mode="File",
            s3_data_distribution_type="ShardedByS3Key",
            
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data", 
            source="/opt/ml/processing/output/train",
            destination=train_path,
            s3_upload_mode="EndOfJob",
        ),
        ProcessingOutput(
            output_name="test_data", 
            source="/opt/ml/processing/output/test",
            destination=test_path,
            s3_upload_mode="EndOfJob",
        ),
    ]
)
train_data_uri = train_path
test_data_uri = test_path

# Specify the output location for the model
model_output_uri = f"s3://{bucket}/{prefix}/model"

# Define hyperparameters
#n_estimators_param = ParameterInteger(name="NEstimators", default_value=50)
#max_depth_param = ParameterInteger(name="MaxDepth", default_value=5)

# Create the SKLearn estimator for training
train_input = TrainingInput(s3_data=train_data_uri, content_type='text/csv')
test_input = TrainingInput(s3_data=test_data_uri, content_type='text/csv')

estimator = SKLearn(entry_point='train.py',
                    framework_version="0.23-1",
                    py_version='py3',
                    instance_type='ml.m5.xlarge',
                    role=role,
                    output_path=model_output_uri,
                    base_job_name='sklearn-iris',
                    hyperparameters={'n_estimators': 50, 'max_depth': 5})
# Define the input data for training and testing
train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "training": train_input,
        "testing": test_input
    },
)
train_step.add_depends_on([processing_step])



# Step 2: Create a pipeline
pipeline = Pipeline(
    name="testingsklearn1",
    steps=[processing_step, train_step],
    sagemaker_session=sagemaker_session,
)
"""
    
    
model = Model(
    image_uri='',  # Specify the container image URI for SKLearn
    model_data=model_output_uri,
    role=role,
    entry_point='inference.py'  # Path to your inference script
)

model_step = ModelStep(
    name='ModelStep',
    model=model,
    inputs={
        'model_data': model_output_uri,
    }
)

# Define the endpoint config and endpoint step
endpoint_config = EndpointConfig(
    name='EndpointConfig',
    model_name=model_step.properties.ModelName,
    instance_type='ml.m5.large'
)

endpoint_step = Endpoint(
    name='EndpointStep',
    endpoint_config_name=endpoint_config.name
)
"""
# Step 3: Submit the pipeline
pipeline.upsert(role_arn=role)

# Start the pipeline execution
execution = pipeline.start()
execution.wait()
