# 1. Import libraries
import os
import json
import boto3

# 2. Knowledge base - Foundation Model & Client SetUp
service_name = 'bedrock-agent-runtime'
client = boto3.client(service_name)


knowledgeBaseID = os.environ['KNOWLEDGE_BASE_ID']
fundation_model_ARN = os.environ['FM_ARN']

# 3. Define the lambda function
def lambda_handler(event, context):
    
    # 3.1. Retrieve User query/Question  
    user_query=event['query']
    
    # 3.2. API Call to "retrieve_and_generate" function
    client_knowledgebase = client.retrieve_and_generate(
    input={
        'text': user_query
    },
    retrieveAndGenerateConfiguration={
        'type': 'KNOWLEDGE_BASE',
        'knowledgeBaseConfiguration': {
            'knowledgeBaseId': knowledgeBaseID,
            'modelArn': fundation_model_ARN
                }
            }
        
    )
            
    # 3.3. Citations - Reference & Final response
    print("----------- Reference Details -------------")

    citations = client_knowledgebase['citations']
    reference = citations[0]['retrievedReferences'][0]

    s3_location = reference['location']['s3Location']['uri']

    generated_response = client_knowledgebase['output']['text']

    # 3.4 Final object to return
    final_result = {
        'statusCode': 200,
        'query': user_query,
        'generated_response': generated_response,
        's3_location': s3_location
    }
    
    # 3.5 Print & Return result
    print("Result details:\n", final_result)
    
    return final_result
        
