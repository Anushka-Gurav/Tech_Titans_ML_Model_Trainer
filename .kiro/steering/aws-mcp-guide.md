---
inclusion: always
---

# AWS MCP Server Configuration Guide

This document provides configuration and usage patterns for AWS MCP server integration with the ML Training Platform project, enabling cloud file operations and command execution.

## AWS MCP Server Overview

The AWS MCP server provides tools for:

- **File Operations**: S3 bucket management, file uploads/downloads
- **Command Execution**: EC2 instance management, Lambda functions
- **Resource Management**: CloudFormation, IAM, monitoring
- **Database Operations**: RDS, DynamoDB interactions

## MCP Configuration

### 1. Basic AWS MCP Server Setup

Add to your `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "aws-mcp": {
      "command": "uvx",
      "args": ["mcp-server-aws@latest"],
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "${AWS_ACCESS_KEY_ID}",
        "AWS_SECRET_ACCESS_KEY": "${AWS_SECRET_ACCESS_KEY}",
        "FASTMCP_LOG_LEVEL": "INFO"
      },
      "disabled": false,
      "autoApprove": [
        "s3_list_objects",
        "s3_get_object",
        "s3_put_object",
        "ec2_describe_instances",
        "lambda_list_functions"
      ]
    }
  }
}
```

### 2. ML Platform Specific Configuration

For the ML Training Platform, configure specific AWS services:

```json
{
  "mcpServers": {
    "aws-ml-platform": {
      "command": "uvx",
      "args": ["mcp-server-aws@latest"],
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "${AWS_ACCESS_KEY_ID}",
        "AWS_SECRET_ACCESS_KEY": "${AWS_SECRET_ACCESS_KEY}",
        "ML_BUCKET_NAME": "ml-training-platform-data",
        "MODEL_BUCKET_NAME": "ml-training-platform-models",
        "LAMBDA_FUNCTION_PREFIX": "ml-platform",
        "FASTMCP_LOG_LEVEL": "INFO"
      },
      "disabled": false,
      "autoApprove": [
        "s3_list_objects",
        "s3_get_object",
        "s3_put_object",
        "s3_delete_object",
        "lambda_invoke",
        "ec2_describe_instances",
        "cloudwatch_get_metric_statistics"
      ]
    }
  }
}
```

## AWS MCP Use Cases for ML Platform

### 1. Dataset Storage and Management

```python
# Example: Upload dataset to S3 using AWS MCP
def upload_dataset_to_s3(dataset_path: str, dataset_id: str) -> str:
    """Upload dataset to S3 bucket using AWS MCP server."""

    # Use AWS MCP to upload file
    s3_key = f"datasets/{dataset_id}/{os.path.basename(dataset_path)}"

    # MCP call would be handled by the agent
    # This is the conceptual flow:
    # aws_mcp.s3_put_object(
    #     bucket="ml-training-platform-data",
    #     key=s3_key,
    #     file_path=dataset_path
    # )

    return f"s3://ml-training-platform-data/{s3_key}"

# Example: List available datasets
def list_datasets_from_s3() -> List[Dict[str, Any]]:
    """List all datasets stored in S3."""

    # MCP call to list objects
    # objects = aws_mcp.s3_list_objects(
    #     bucket="ml-training-platform-data",
    #     prefix="datasets/"
    # )

    datasets = []
    # Process objects and return dataset metadata
    return datasets
```

### 2. Model Storage and Versioning

```python
# Example: Store trained models in S3
def store_model_in_s3(model_path: str, model_metadata: Dict[str, Any]) -> str:
    """Store trained model and metadata in S3."""

    model_id = model_metadata["id"]
    timestamp = datetime.now().isoformat()

    # Store model file
    model_key = f"models/{model_id}/model_{timestamp}.pkl"

    # Store metadata
    metadata_key = f"models/{model_id}/metadata_{timestamp}.json"

    # MCP calls for file operations
    # aws_mcp.s3_put_object(
    #     bucket="ml-training-platform-models",
    #     key=model_key,
    #     file_path=model_path
    # )

    # aws_mcp.s3_put_object(
    #     bucket="ml-training-platform-models",
    #     key=metadata_key,
    #     content=json.dumps(model_metadata)
    # )

    return f"s3://ml-training-platform-models/{model_key}"
```

### 3. Distributed Training with EC2

```python
# Example: Launch EC2 instances for distributed training
def launch_training_cluster(training_config: Dict[str, Any]) -> List[str]:
    """Launch EC2 instances for distributed model training."""

    instance_config = {
        "ImageId": "ami-0abcdef1234567890",  # ML-optimized AMI
        "InstanceType": "ml.m5.xlarge",
        "MinCount": training_config.get("num_instances", 1),
        "MaxCount": training_config.get("num_instances", 1),
        "KeyName": "ml-platform-key",
        "SecurityGroupIds": ["sg-ml-training"],
        "UserData": base64.b64encode(get_training_script().encode()).decode()
    }

    # MCP call to launch instances
    # instances = aws_mcp.ec2_run_instances(**instance_config)

    instance_ids = []  # Extract from response
    return instance_ids

def monitor_training_progress(instance_ids: List[str]) -> Dict[str, Any]:
    """Monitor training progress on EC2 instances."""

    progress = {}
    for instance_id in instance_ids:
        # MCP call to get instance status
        # status = aws_mcp.ec2_describe_instances(instance_ids=[instance_id])

        # MCP call to get CloudWatch metrics
        # metrics = aws_mcp.cloudwatch_get_metric_statistics(
        #     namespace="ML/Training",
        #     metric_name="TrainingProgress",
        #     dimensions=[{"Name": "InstanceId", "Value": instance_id}]
        # )

        progress[instance_id] = {
            "status": "running",  # Extract from status
            "progress": 0.75,     # Extract from metrics
            "eta": "5 minutes"    # Calculate from metrics
        }

    return progress
```

### 4. Serverless Training with Lambda

```python
# Example: Use Lambda for lightweight training tasks
def trigger_lambda_training(dataset_s3_path: str, algorithm_config: Dict[str, Any]) -> str:
    """Trigger Lambda function for model training."""

    payload = {
        "dataset_path": dataset_s3_path,
        "algorithm": algorithm_config["algorithm"],
        "hyperparameters": algorithm_config["hyperparameters"],
        "output_bucket": "ml-training-platform-models"
    }

    # MCP call to invoke Lambda function
    # response = aws_mcp.lambda_invoke(
    #     function_name="ml-platform-train-model",
    #     payload=json.dumps(payload),
    #     invocation_type="Event"  # Async execution
    # )

    execution_id = "lambda-execution-123"  # Extract from response
    return execution_id

def check_lambda_training_status(execution_id: str) -> Dict[str, Any]:
    """Check status of Lambda training execution."""

    # MCP call to get Lambda logs
    # logs = aws_mcp.cloudwatch_get_log_events(
    #     log_group_name="/aws/lambda/ml-platform-train-model",
    #     log_stream_name=execution_id
    # )

    return {
        "status": "completed",
        "model_s3_path": "s3://ml-training-platform-models/model_123.pkl",
        "metrics": {"accuracy": 0.95, "f1_score": 0.92}
    }
```

## AWS MCP Integration Patterns

### 1. File Operations Pattern

```python
class AWSFileManager:
    """Manages file operations using AWS MCP server."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    async def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload file to S3 using MCP."""
        try:
            # Agent will use AWS MCP to execute this
            logger.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3 using MCP."""
        try:
            # Agent will use AWS MCP to execute this
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in S3 bucket using MCP."""
        try:
            # Agent will use AWS MCP to execute this
            files = []  # MCP response would populate this
            return files
        except Exception as e:
            logger.error(f"List files failed: {e}")
            return []
```

### 2. Command Execution Pattern

```python
class AWSCommandExecutor:
    """Executes AWS commands using MCP server."""

    async def create_training_environment(self, config: Dict[str, Any]) -> str:
        """Create AWS resources for training environment."""

        # Create S3 buckets if they don't exist
        await self._ensure_bucket_exists("ml-training-data")
        await self._ensure_bucket_exists("ml-training-models")

        # Launch EC2 instances if needed
        if config.get("use_ec2", False):
            instance_ids = await self._launch_training_instances(config)
            return f"Training environment created with instances: {instance_ids}"

        # Set up Lambda functions if needed
        if config.get("use_lambda", False):
            function_arn = await self._deploy_training_lambda(config)
            return f"Training environment created with Lambda: {function_arn}"

        return "Local training environment configured"

    async def _ensure_bucket_exists(self, bucket_name: str) -> bool:
        """Ensure S3 bucket exists using MCP."""
        # Agent will use AWS MCP to check and create bucket
        return True

    async def _launch_training_instances(self, config: Dict[str, Any]) -> List[str]:
        """Launch EC2 instances using MCP."""
        # Agent will use AWS MCP to launch instances
        return ["i-1234567890abcdef0"]

    async def _deploy_training_lambda(self, config: Dict[str, Any]) -> str:
        """Deploy Lambda function using MCP."""
        # Agent will use AWS MCP to deploy function
        return "arn:aws:lambda:us-east-1:123456789012:function:ml-training"
```

## Environment Variables Setup

Create `.env.example` file for AWS configuration:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# ML Platform S3 Buckets
ML_DATA_BUCKET=ml-training-platform-data
ML_MODELS_BUCKET=ml-training-platform-models

# EC2 Configuration
EC2_KEY_NAME=ml-platform-key
EC2_SECURITY_GROUP=sg-ml-training
EC2_INSTANCE_TYPE=ml.m5.xlarge

# Lambda Configuration
LAMBDA_FUNCTION_PREFIX=ml-platform
LAMBDA_RUNTIME=python3.9
LAMBDA_TIMEOUT=900
```

## Security Best Practices

### 1. IAM Permissions

Create minimal IAM policy for ML Platform:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ml-training-platform-*",
        "arn:aws:s3:::ml-training-platform-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:DescribeInstances",
        "ec2:TerminateInstances"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "ec2:InstanceType": ["ml.m5.xlarge", "ml.m5.large"]
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:InvokeFunction",
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode"
      ],
      "Resource": "arn:aws:lambda:*:*:function:ml-platform-*"
    }
  ]
}
```

### 2. Credential Management

- Use AWS IAM roles when possible
- Store credentials in environment variables, not code
- Rotate access keys regularly
- Use AWS Secrets Manager for sensitive data

## Testing AWS MCP Integration

### 1. Connection Test

```python
async def test_aws_mcp_connection():
    """Test AWS MCP server connection."""
    try:
        # Test S3 access
        buckets = await list_s3_buckets()
        logger.info(f"Connected to AWS. Found {len(buckets)} buckets")

        # Test EC2 access
        instances = await list_ec2_instances()
        logger.info(f"EC2 access confirmed. Found {len(instances)} instances")

        return True
    except Exception as e:
        logger.error(f"AWS MCP connection failed: {e}")
        return False
```

### 2. Integration Test

```python
async def test_ml_platform_aws_integration():
    """Test ML Platform AWS integration."""

    # Test dataset upload
    test_dataset = "test_data.csv"
    upload_success = await upload_dataset_to_s3(test_dataset, "test-dataset-001")
    assert upload_success, "Dataset upload failed"

    # Test model storage
    test_model = "test_model.pkl"
    model_path = await store_model_in_s3(test_model, {"id": "test-model-001"})
    assert model_path.startswith("s3://"), "Model storage failed"

    # Test training environment
    env_id = await create_training_environment({"use_lambda": True})
    assert "Lambda" in env_id, "Training environment creation failed"

    logger.info("✅ All AWS MCP integration tests passed")
```

## Usage in ML Training Platform

### 1. Dataset Management Service

```python
class CloudDatasetManager(DatasetManager):
    """Extended dataset manager with AWS cloud storage."""

    def __init__(self):
        super().__init__()
        self.aws_file_manager = AWSFileManager("ml-training-platform-data")

    async def upload_dataset(self, file_path: str) -> Dataset:
        """Upload dataset with cloud backup."""

        # Local processing first
        dataset = super().upload_dataset(file_path)

        # Backup to S3
        s3_key = f"datasets/{dataset.id}/{dataset.filename}"
        await self.aws_file_manager.upload_file(file_path, s3_key)

        # Update dataset with cloud path
        dataset.cloud_path = f"s3://ml-training-platform-data/{s3_key}"

        return dataset
```

### 2. Distributed Training Service

```python
class CloudModelTrainer(ModelTrainer):
    """Extended model trainer with cloud training capabilities."""

    async def train_model_distributed(self, dataset_id: str, algorithm: str,
                                    hyperparams: Dict[str, Any]) -> TrainedModel:
        """Train model using distributed AWS resources."""

        # Check if distributed training is beneficial
        dataset = self.get_dataset(dataset_id)
        if dataset.row_count > 100000:  # Large dataset
            return await self._train_on_ec2(dataset, algorithm, hyperparams)
        else:
            return await self._train_on_lambda(dataset, algorithm, hyperparams)

    async def _train_on_ec2(self, dataset: Dataset, algorithm: str,
                          hyperparams: Dict[str, Any]) -> TrainedModel:
        """Train on EC2 instances for large datasets."""

        # Launch training cluster
        cluster_id = await launch_training_cluster({
            "algorithm": algorithm,
            "hyperparams": hyperparams,
            "dataset_path": dataset.cloud_path
        })

        # Monitor training progress
        while True:
            progress = await monitor_training_progress([cluster_id])
            if progress[cluster_id]["status"] == "completed":
                break
            await asyncio.sleep(30)

        # Retrieve trained model
        model_path = progress[cluster_id]["model_path"]
        return self._create_model_from_s3_path(model_path)
```

This AWS MCP integration enables the ML Training Platform to leverage cloud resources for scalable dataset storage, distributed training, and model management while maintaining the same local development experience.
