use crate::immutable_agent::Message;
use crate::llama_structs::{ output_llama_response, Content, LlamaResponseMessage };
use endpoints::*;
// use endpoints::{
//     chat,
//     ChatCompletionRequestBuilder,
//     ChatCompletionRequest,
//     CompletionRequest,
//     ChatCompletionRequestMessage,
//     CompletionUsage,
//     CompletionObject,
//     CompletionChoice,
//     CompletionRequestBuilder,
//     FinishReason,
//     Usage,
//     ChatCompletionObjectMessage,
//     ChatCompletionAssistantMessage,
//     ChatCompletionUserMessage,
//     ToolChoiceTool,
//     ChatCompletionUserMessageContent,
//     Fuction,
//     ToolCall,
//     ChatCompletionToolMessage,
//     ChatCompletionObject,
//     JSONSchemaDefine,
//     ChatCompletionRequestFunctionParameters,
//     ChatCompletionRequestFunction,
//     TextContentPart,
//     ChatCompletionUserMessageContent,
// };
use chat_prompts::*;
use serde::Deserialize;
use std::collections::HashMap;

pub async fn chat_inner_async(
    system_prompt: &str,
    user_input: &str,
    max_token: u16
) -> anyhow::Result<CreateChatCompletionResponse> {
    let mut headers = HeaderMap::new();
    let api_key = std::env::var("LLAMA_API_KEY").expect("LLAMA_API_KEY must be set");
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    let config = LocalServiceProviderConfig {
        api_base: String::from("http://127.0.0.1:8080/v1"),
        headers: headers,
        api_key: Secret::new(api_key),
        query: HashMap::new(),
    };

    // stop: ['</s>', '[/INST]'],
    let model = "Hermes-2-Pro-Llama-3-8B";
    let client = OpenAIClient::with_config(config);
    let messages = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()
            .expect("Failed to build system message")
            .into(),
        ChatCompletionRequestUserMessageArgs::default().content(user_input).build()?.into()
    ];
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(max_token)
        .model(model)
        .messages(messages)
        .build()?;

    let chat = match client.chat().create(request).await {
        Ok(chat) => chat,
        Err(_e) => {
            println!("Error getting response from OpenAI: {:?}", _e);
            return Err(anyhow::anyhow!("Failed to get reply from OpenAI: {:?}", _e));
        }
    };

    Ok(chat.clone())
}

impl Message {
    pub fn content_to_string(&self) -> String {
        match &self.content {
            Content::Text(text) => text.clone(),
            Content::ToolCall(tool_call) =>
                format!(
                    "tool_call: {}, arguments: {}",
                    tool_call.name,
                    tool_call.arguments
                        .as_ref()
                        .unwrap()
                        .into_iter()
                        .map(|(arg, val)| format!("{:?}: {:?}", arg, val))
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
        }
    }
}

#[allow(deprecated)]
impl From<Message> for ChatCompletionRequestMessage {
    fn from(message: Message) -> ChatCompletionRequestMessage {
        match message.role {
            Role::System => {
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: message.content_to_string(),
                    role: Role::System,
                    name: message.name,
                })
            }
            Role::User =>
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(
                        message.content_to_string()
                    ),
                    role: Role::User,
                    name: message.name,
                }),
            Role::Assistant => {
                ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                    content: Some(message.content_to_string()),
                    role: Role::Assistant,
                    name: message.name,
                    tool_calls: None,
                    function_call: None,
                })
            }
            _ =>
                ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                    content: Some(message.content_to_string()),
                    role: Role::Assistant,
                    name: message.name,
                    tool_calls: None,
                    function_call: None,
                }),
        }
    }
}

pub async fn chat_inner_async_llama(
    messages: Vec<Message>,
    max_token: u16
) -> anyhow::Result<LlamaResponseMessage> {
    let mut headers = HeaderMap::new();
    let api_key = std::env::var("LLAMA_API_KEY").expect("LLAMA_API_KEY must be set");
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    let config = LocalServiceProviderConfig {
        api_base: String::from("http://127.0.0.1:8080/v1"),
        headers: headers,
        api_key: Secret::new(api_key),
        query: HashMap::new(),
    };

    // stop: ['</s>', '[/INST]'],
    let model = "Hermes-2-Pro-Llama-3-8B";
    let client = OpenAIClient::with_config(config);

    let messages: Vec<ChatCompletionRequestMessage> = messages
        .into_iter()
        .map(ChatCompletionRequestMessage::from)
        .collect();

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(max_token)
        .model(model)
        .messages(messages)
        .build()?;

    match client.chat().create(request).await {
        Ok(chat) => {
            if let Some(out) = output_llama_response(chat) {
                Ok(out)
            } else {
                Err(anyhow::anyhow!("Empty output in Llama format"))
            }
        }
        Err(_e) => {
            println!("Error getting response from OpenAI: {:?}", _e);
            Err(anyhow::anyhow!("Failed to get reply from OpenAI: {:?}", _e))
        }
    }
}

pub fn parse_summary_from_raw_json(input: &str) -> String {
    #[derive(Deserialize, Debug)]
    struct SummaryStruct {
        impactful: Option<String>,
        alignment: Option<String>,
        patterns: Option<String>,
        synergy: Option<String>,
        significance: Option<String>,
    }

    let summary: SummaryStruct = serde_json::from_str(input).expect("Failed to parse summary JSON");

    let fields = [
        &summary.impactful,
        &summary.alignment,
        &summary.patterns,
        &summary.synergy,
        &summary.significance,
    ];

    fields
        .iter()
        .filter_map(|&field| field.as_ref()) // Convert Option<&String> to Option<&str>
        .filter(|field| !field.is_empty()) // Filter out empty strings
        .fold(String::new(), |mut acc, field| {
            if !acc.is_empty() {
                acc.push_str(" ");
            }
            acc.push_str(field);
            acc
        })
}

pub fn parse_issue_summary_from_json(input: &str) -> anyhow::Result<Vec<(String, String)>> {
    let parsed: serde_json::Map<String, serde_json::Value> = serde_json::from_str(input)?;

    let summaries = parsed
        .iter()
        .filter_map(|(key, value)| {
            if let Some(summary_str) = value.as_str() {
                Some((key.clone(), summary_str.to_owned()))
            } else {
                None
            }
        })
        .collect::<Vec<(String, String)>>(); // Collect into a Vec of tuples

    Ok(summaries)
}
