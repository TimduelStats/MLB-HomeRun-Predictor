#!/usr/bin/env python3
import aws_cdk as cdk
from infra_stack import InfraStack

app = cdk.App()
InfraStack(app, "MLBHomeRunPredictorInfraStack")
app.synth() #  It takes the Python code in InfraStack and converts it into a CloudFormation template