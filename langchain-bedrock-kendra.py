import json
import boto3
import time
from typing import Any, List, Mapping, Optional
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, DynamoDBChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import logging
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import AIMessage, BaseMessage, ChatResult, HumanMessage, SystemMessage, ChatGeneration, LLMResult
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict
from boto3.dynamodb.conditions import Key

MAX_CONTENT_LENGTH = 2000
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_runtime = boto3.client("bedrock-runtime")
kendra = boto3.client('kendra')
dynamodb = boto3.client('dynamodb')

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Custom ChatModel for Bedrock
class BedrockChatModel(BaseChatModel):
    model_id: str
    client: Any
    guardrails: Optional[str] = None
    
    @property
    def _llm_type(self) -> str:
        return "bedrock-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        bedrock_messages = self._convert_messages_to_bedrock_format(messages)
        
        logger.info(f"Generated prompt for Bedrock: {json.dumps(bedrock_messages, indent=2)}")
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": bedrock_messages,
            "temperature": 0.7,
        }

        if self.guardrails:
            request_body["system"] = self.guardrails

        if stop:
            request_body["stop_sequences"] = stop

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        full_response = ""
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            if chunk['type'] == 'content_block_delta':
                full_response += chunk['delta']['text']
                if run_manager:
                    run_manager.on_llm_new_token(chunk['delta']['text'])
        
        logger.info(f"Bedrock model response: {full_response}")
        
        ai_message = AIMessage(content=full_response)
        chat_generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[chat_generation])

    def _convert_messages_to_bedrock_format(self, messages: List[BaseMessage]) -> List[dict]:
        bedrock_messages = []
        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                if i == 0 or not bedrock_messages:
                    bedrock_messages.append({"role": "user", "content": f"System: {message.content}"})
                else:
                    bedrock_messages[-1]["content"] += f"\n\nSystem: {message.content}"
            elif isinstance(message, HumanMessage):
                if bedrock_messages and bedrock_messages[-1]["role"] == "user":
                    bedrock_messages[-1]["content"] += f"\n\nHuman: {message.content}"
                else:
                    bedrock_messages.append({"role": "user", "content": f"Human: {message.content}"})
            elif isinstance(message, AIMessage):
                bedrock_messages.append({"role": "assistant", "content": message.content})
        
        if bedrock_messages and bedrock_messages[0]["role"] == "assistant":
            bedrock_messages.insert(0, {"role": "user", "content": "Please continue."})
        
        return bedrock_messages

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id}

# Modified DynamoDBChatMessageHistory
class DynamoDBChatMessageHistory(DynamoDBChatMessageHistory):
    @property
    def messages(self) -> List[BaseMessage]:
        response = self.table.query(
            KeyConditionExpression=Key("SessionId").eq(self.session_id),
            ScanIndexForward=True,
        )
        items = response["Items"]
        messages = []
        for item in items:
            history = self._parse_history(item.get("History"))
            messages.extend(messages_from_dict(history))
        return messages

    def _parse_history(self, history):
        if isinstance(history, str):
            try:
                return json.loads(history)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse history: {history}")
                return []
        return history

# Define guardrails
guardrails = """
You are an AI assistant for product information. Be concise in your responses.Always be polite and professional.
Never provide information about competitors' Jarvis products.
Do not discuss availability. If asked, say this information changes frequently and encourage users to visit our website or contact customer service for the most up-to-date information.
Always respect user privacy and do not ask for or store personal information.
"""

# Initialize BedrockChatModel with guardrails
llm = BedrockChatModel(model_id=model_id, client=bedrock_runtime, guardrails=guardrails)

kendra_index_id = '59fcb117-0007-4dd7-99a3-3ff6dea8d7d4'
kendra_retriever = AmazonKendraRetriever(index_id=kendra_index_id)

# Create a custom prompt template
system_template = """You are an assistant for question-answering tasks for product information."""
human_template = """Read the following context and answer the question:
Context: {context}

Question: {question}"""

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

def create_qa_chain(memory):
    condense_question_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        HumanMessagePromptTemplate.from_template("Chat History:\n{chat_history}\nFollow Up Question: {question}\nStandalone question:")
    ])
    
    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=chat_prompt)

    return ConversationalRetrievalChain(
        retriever=kendra_retriever,
        memory=memory,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        return_generated_question=True,
        output_key='answer'
    )

def truncate_content(content, max_length):
    if len(content) <= max_length:
        return content
    sentences = content.split('.')
    truncated = ''
    for sentence in sentences:
        if len(truncated) + len(sentence) + 1 > max_length:
            break
        truncated += sentence + '.'
    return truncated.strip()

def handler(event, context):
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        session_id = event.get('sessionId', 'default')
        query = event['inputTranscript']
        
        if not query:
            raise ValueError("No input transcript provided in the event")

        table_name = 'ConversationHistory'

        message_history = DynamoDBChatMessageHistory(
            table_name=table_name,
            session_id=session_id
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            output_key='answer'
        )

        qa_chain = create_qa_chain(memory)

        response = qa_chain({"question": query})
        answer = response['answer']
        source_docs = response.get('source_documents', [])

        formatted_sources = [
            f"Title: {doc.metadata.get('title', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}"
            for doc in source_docs
        ]

        full_response = f"{answer}\n\nSources:\n" + "\n".join(formatted_sources)
        
        logger.info(f"Generated response: {full_response}")

        # Save the answer to memory manually
        current_timestamp = int(time.time() * 1000)  # current time in milliseconds
        dynamodb.put_item(
            TableName=table_name,
            Item={
                'SessionId': {'S': session_id},
                'Timestamp': {'N': str(current_timestamp)},
                'History': {'S': json.dumps([{"type": "human", "data": {"content": query, "additional_kwargs": {}}},
                                             {"type": "ai", "data": {"content": answer, "additional_kwargs": {}}}])}
            }
        )

        truncated_content = truncate_content(full_response, MAX_CONTENT_LENGTH)

        return {
            'sessionState': {
                'dialogAction': {
                    'type': 'ElicitIntent'
                },
                'intent': {
                    'name': event['sessionState']['intent']['name'],
                    'state': 'Fulfilled'
                }
            },
            'messages': [
                {
                    'contentType': 'PlainText',
                    'content': truncated_content
                }
            ]
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return {
            'sessionState': {
                'dialogAction': {
                    'type': 'ElicitIntent'
                },
                'intent': {
                    'name': event['sessionState']['intent']['name'],
                    'state': 'Failed'
                }
            },
            'messages': [
                {
                    'contentType': 'PlainText',
                    'content': "I'm sorry, but I encountered an error. Please try again later."
                }
            ]
        }
