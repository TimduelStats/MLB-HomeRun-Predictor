from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigateway as apigateway,
    Duration,
    Stack
)
from constructs import Construct

class InfraStack(Stack):
    ## which command build push the image
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # This creates a Lambda function that runs a Docker image.
        lambda_function = lambda_.DockerImageFunction(
            self, "mlb-homerun-prediction",
            # Build a Docker image from the Dockerfile located in the ../src directory.
            # Push the built image to Amazon Elastic Container Registry (ECR).
            code=lambda_.DockerImageCode.from_image_asset("../src"),
            timeout=Duration.seconds(300),
            memory_size=3000,
        )

        # Todo: no usage for below code
        api = apigateway.LambdaRestApi(
            self, "Endpoint",
            handler=lambda_function,
            proxy=False
        )

        integration = apigateway.LambdaIntegration(lambda_function)
        api.root.add_method('GET', integration, api_key_required=True)

        plan = api.add_usage_plan(
            'UsagePlan',
            name='Basic',
            quota=apigateway.QuotaSettings(
                limit=1000,
                period=apigateway.Period.DAY
            )
        )

        key = api.add_api_key('ApiKey')
        plan.add_api_key(key)
        plan.add_api_stage(
            stage=api.deployment_stage
        )